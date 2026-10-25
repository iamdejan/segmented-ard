"""Centered Kernel Alignment evaluation for teacher and student encoders.

This script mirrors the BDD100k loading and checkpoint conventions used by
``evaluate_performance.py`` and compares the internal encoder representations of
a teacher model and a student model with linear Centered Kernel Alignment (CKA).

The compared representations come from explicitly named encoder modules, which
makes every CKA row and column traceable to the exact module output that
generated it. Each selected output is globally average pooled from
``(B, C, H, W)`` to ``(B, C)`` before comparison. The implementation
accumulates sufficient statistics batch-by-batch, so it does not keep the
complete activation dataset or any ``N x N`` Gram matrix in memory.
"""

import argparse
import os
from typing import Callable, Dict, List, Sequence, Tuple, cast

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from torchinfo import summary

from evaluate_performance import (
    BDDSegmentationDataset,
    BatchImage,
    Configuration,
    ImageTensor,
    MaskTensor,
    load_dataset_from_files,
    load_trained_model,
)


# CKA operates on one vector per image and selected module output. The anonymous
# feature axis allows teacher and student layers to have different channel
# widths while the named batch axis still verifies corresponding image rows.
PooledRepresentation = Float[Tensor, "b _feature"]
CKAMatrix = Float[np.ndarray, "teacher_layer student_layer"]
FeatureMapTensor = Float[Tensor, "b c h w"]
ActivationStore = Dict[str, List[Tensor]]
ForwardHook = Callable[[nn.Module, Tuple[object, ...], object], None]


# These defaults identify the exact encoder modules whose *outputs* correspond
# to the five multi-resolution feature maps used by the U-Net decoder. Keeping
# the names explicit makes the exported CKA matrix auditable instead of relying
# on anonymous stage positions such as T1/S1.
DEFAULT_TEACHER_LAYERS: Tuple[str, ...] = (
    "relu",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
)
DEFAULT_STUDENT_LAYERS: Tuple[str, ...] = (
    "features.1",
    "features.3",
    "features.6",
    "features.13",
    "features.18",
)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when evaluation uses ``nn.DataParallel``.

    The serialized checkpoints may contain either a plain module or a
    ``DataParallel`` wrapper. CKA needs direct access to the segmentation
    model's ``encoder`` attribute, so this helper normalizes both cases before
    feature extraction.

    Steps
    -----
    1. Check whether ``model`` is an ``nn.DataParallel`` instance.
    2. Return ``model.module`` for wrapped checkpoints, otherwise return the
       original module unchanged.

    Parameters
    ----------
    model : nn.Module
        Loaded teacher or student segmentation model.

    Returns
    -------
    nn.Module
        The underlying segmentation model that exposes encoder components.
    """
    # DataParallel stores the user model under ``module``; unwrapping it keeps
    # encoder access consistent with the visualization path in the template.
    return model.module if isinstance(model, nn.DataParallel) else model


def get_encoder(model: nn.Module) -> nn.Module:
    """Return the encoder module from a loaded segmentation model.

    The CKA analysis compares encoder representations only. Centralizing the
    encoder lookup avoids duplicating wrapper handling and gives a clear error
    when a checkpoint does not expose the architecture expected by this script.

    Steps
    -----
    1. Unwrap ``nn.DataParallel`` when present.
    2. Read the model's ``encoder`` attribute.
    3. Validate that the attribute is an ``nn.Module`` before returning it.

    Parameters
    ----------
    model : nn.Module
        Loaded teacher or student segmentation model.

    Returns
    -------
    nn.Module
        Encoder module used for named-layer activation extraction.

    Raises
    ------
    AttributeError
        If the model does not expose an ``nn.Module`` named ``encoder``.
    """
    # CKA layer names are resolved relative to the encoder so the command line
    # stays concise (for example ``layer3`` rather than ``encoder.layer3``).
    base_model = unwrap_model(model)
    encoder = getattr(base_model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise AttributeError(
            f"{base_model.__class__.__name__} does not expose an nn.Module "
            "'encoder' attribute required for named-layer CKA."
        )
    return encoder


def normalize_encoder_layer_names(layer_names: Sequence[str]) -> List[str]:
    """Normalize requested layer names to encoder-relative module paths.

    Users may provide either ``layer3`` or ``encoder.layer3``. Internally the
    hooks are registered on ``model.encoder``, so the optional ``encoder.``
    prefix is removed while preserving the requested ordering.

    Steps
    -----
    1. Strip whitespace from every requested layer path.
    2. Remove a leading ``encoder.`` prefix when present.
    3. Reject empty or duplicated paths because they would make matrix labels
       ambiguous.

    Parameters
    ----------
    layer_names : Sequence[str]
        Requested module paths, either encoder-relative or prefixed by
        ``encoder.``.

    Returns
    -------
    List[str]
        Ordered encoder-relative module paths suitable for
        ``encoder.named_modules()`` lookup.

    Raises
    ------
    ValueError
        If no layer names are supplied, a path is empty, or a path is repeated.
    """
    normalized: List[str] = []
    for layer_name in layer_names:
        # Accepting the full ``encoder.*`` spelling makes CLI arguments match
        # the labels shown in the final heatmap without changing hook lookup.
        name = layer_name.strip()
        if name.startswith("encoder."):
            name = name[len("encoder.") :]
        if not name:
            raise ValueError("CKA layer names must not be empty.")
        normalized.append(name)

    if not normalized:
        raise ValueError("At least one encoder layer must be selected for CKA.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"CKA layer names must be unique, got {normalized}.")
    return normalized


def make_activation_hook(
    layer_name: str,
    activation_store: ActivationStore,
    model_label: str,
) -> ForwardHook:
    """Create a forward hook that records one named module's tensor output.

    Recording outputs through hooks follows the same layer-selection principle
    used by common PyTorch CKA libraries: a matrix coordinate refers directly
    to a named module output rather than to an inferred stage index. Outputs are
    appended instead of overwritten so reused modules can be detected later.

    Steps
    -----
    1. Receive the selected module's forward output.
    2. Verify that the output is a tensor suitable for convolutional CKA.
    3. Append it to the per-layer activation list for this forward pass.

    Parameters
    ----------
    layer_name : str
        Encoder-relative module path being observed.
    activation_store : ActivationStore
        Mutable mapping from module path to outputs captured in the current
        encoder forward pass.
    model_label : str
        Human-readable model label used in validation errors.

    Returns
    -------
    ForwardHook
        Callable compatible with ``nn.Module.register_forward_hook``.

    Raises
    ------
    TypeError
        If the selected module returns a non-tensor output.
    """

    def capture_output(
        _module: nn.Module,
        _inputs: Tuple[object, ...],
        output: object,
    ) -> None:
        # CKA expects one tensor representation from each named endpoint. A
        # tuple/dict output would require an explicit policy for which element
        # represents the layer, so fail instead of selecting one silently.
        if not isinstance(output, Tensor):
            raise TypeError(
                f"{model_label} layer 'encoder.{layer_name}' returned "
                f"{type(output)}, but CKA requires a Tensor output."
            )
        activation_store[layer_name].append(output)

    return capture_output


def register_named_activation_hooks(
    encoder: nn.Module,
    layer_names: Sequence[str],
    activation_store: ActivationStore,
    model_label: str,
) -> List[RemovableHandle]:
    """Register forward hooks on explicitly selected encoder modules.

    The requested paths are resolved against ``encoder.named_modules()`` before
    any evaluation begins. This guarantees that every heatmap label names a
    real module in the loaded checkpoint and prevents silent fallback to a
    different architectural stage.

    Steps
    -----
    1. Build a lookup from ``encoder.named_modules()``.
    2. Validate that every requested path exists.
    3. Initialize one activation list per selected path.
    4. Register a forward hook on each selected module in the requested order.

    Parameters
    ----------
    encoder : nn.Module
        Teacher or student encoder on which hooks are registered.
    layer_names : Sequence[str]
        Ordered encoder-relative module paths to capture.
    activation_store : ActivationStore
        Mutable mapping populated by the registered hooks.
    model_label : str
        Human-readable model label used in validation errors.

    Returns
    -------
    List[RemovableHandle]
        Hook handles that must be removed after CKA evaluation.

    Raises
    ------
    ValueError
        If a requested module path does not exist in the encoder.
    """
    named_modules = dict(encoder.named_modules())
    missing_layers = [name for name in layer_names if name not in named_modules]
    if missing_layers:
        raise ValueError(
            f"{model_label} encoder does not contain requested CKA layer(s): "
            f"{missing_layers}. Use names from model.encoder.named_modules()."
        )

    handles: List[RemovableHandle] = []
    for layer_name in layer_names:
        # Pre-create the list so the hook can append without mutating the set of
        # mapping keys during an encoder forward pass.
        activation_store[layer_name] = []
        hook = make_activation_hook(layer_name, activation_store, model_label)
        handles.append(named_modules[layer_name].register_forward_hook(hook))
    return handles


@jaxtyped(typechecker=beartype)
def global_average_pool(feature_map: FeatureMapTensor) -> PooledRepresentation:
    """Collapse a convolutional feature map to one channel vector per image.

    CKA should compare compact named-layer representations rather than flattened
    spatial maps, which would make early high-resolution layers prohibitively
    expensive. Global average pooling preserves each channel's aggregate
    response while removing the spatial axes.

    Steps
    -----
    1. Average every channel over its complete spatial extent.
    2. Flatten the two singleton spatial dimensions to obtain ``(B, C)``.

    Parameters
    ----------
    feature_map : Float[Tensor, "b c h w"]
        Selected encoder-module activation tensor with shape ``(B, C, H, W)``.

    Returns
    -------
    PooledRepresentation
        Per-image representation matrix with shape ``(B, C)``.
    """
    # Adaptive pooling avoids assuming that the selected ResNet-18 and
    # MobileNetV2 outputs have matching spatial resolutions.
    return cast(
        PooledRepresentation,
        F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(start_dim=1),
    )


@jaxtyped(typechecker=beartype)
def extract_named_encoder_representations(
    encoder: nn.Module,
    x: BatchImage,
    layer_names: Sequence[str],
    activation_store: ActivationStore,
    model_label: str,
) -> Tuple[List[PooledRepresentation], List[str]]:
    """Extract globally pooled outputs from explicitly named encoder modules.

    Forward hooks identify the exact module that produced every compared
    representation. The encoder is still run only once per batch, and each
    captured feature map is globally average pooled immediately so the
    streaming CKA calculation remains memory-efficient.

    Steps
    -----
    1. Reset the hook store for the selected module names.
    2. Run the encoder once so every selected forward hook captures its output.
    3. Require each selected module to execute exactly once, avoiding ambiguous
       outputs from reused modules.
    4. Global-average-pool each captured ``(B, C, H, W)`` tensor to ``(B, C)``.
    5. Build labels containing the exact module path, channel count, and spatial
       size, then release references to the large feature maps.

    Parameters
    ----------
    encoder : nn.Module
        Encoder whose selected named modules already have forward hooks.
    x : BatchImage
        Input image batch with shape ``(B, 3, H, W)``.
    layer_names : Sequence[str]
        Ordered encoder-relative module paths whose outputs are compared.
    activation_store : ActivationStore
        Hook-populated mapping from module path to captured outputs.
    model_label : str
        Human-readable model label used in validation errors.

    Returns
    -------
    Tuple[List[PooledRepresentation], List[str]]
        Pooled named-module representations and matching traceable labels.

    Raises
    ------
    ValueError
        If a selected module is not executed exactly once or its output does
        not use the expected ``(B, C, H, W)`` layout.
    """
    # Reset existing references before the forward pass so the validation below
    # describes exactly this batch rather than activations from an earlier one.
    for layer_name in layer_names:
        activation_store[layer_name] = []

    # The encoder's returned feature sequence is intentionally ignored here:
    # named forward hooks, not positional stage outputs, define the CKA inputs.
    encoder(x)

    representations: List[PooledRepresentation] = []
    labels: List[str] = []
    try:
        for layer_name in layer_names:
            captured_outputs = activation_store[layer_name]
            if len(captured_outputs) != 1:
                raise ValueError(
                    f"{model_label} layer 'encoder.{layer_name}' executed "
                    f"{len(captured_outputs)} times in one encoder forward; CKA "
                    "requires an unambiguous single output per selected layer."
                )

            feature = captured_outputs[0]
            if feature.ndim != 4:
                raise ValueError(
                    f"{model_label} layer 'encoder.{layer_name}' has output "
                    f"shape {tuple(feature.shape)}; CKA expects (B, C, H, W)."
                )

            # Pool before accumulation so early high-resolution feature maps do
            # not make CKA memory/computation scale with H x W.
            pooled = global_average_pool(cast(FeatureMapTensor, feature))
            representations.append(pooled)

            _, channels, height, width = feature.shape
            labels.append(
                f"encoder.{layer_name}\nC={channels}, {height}x{width}"
            )
    finally:
        # Hooks otherwise keep references to full-resolution activations. Clear
        # them immediately after pooling so only the compact (B, C) tensors live
        # through the dataset-wide streaming calculation.
        for layer_name in layer_names:
            activation_store[layer_name] = []

    return representations, labels


@jaxtyped(typechecker=beartype)
def compute_linear_cka_matrix(
    teacher: nn.Module,
    student: nn.Module,
    teacher_layer_names: Sequence[str],
    student_layer_names: Sequence[str],
    dataloader: DataLoader[Tuple[ImageTensor, MaskTensor]],
    device: torch.device,
) -> Tuple[CKAMatrix, List[str], List[str], int]:
    """Compute named-layer linear CKA using streaming sufficient statistics.

    For centered representation matrices ``X`` and ``Y``, linear CKA is
    ``||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F)``. Instead of
    storing every pooled activation, this function accumulates the sums and
    second moments needed to construct the globally centered cross-products at
    the end. Explicit forward hooks ensure each row/column remains associated
    with an exact encoder module output, while global average pooling keeps
    memory dependent on channel widths rather than on spatial resolution or the
    number of evaluation images.

    Steps
    -----
    1. Normalize and validate the requested teacher/student module paths.
    2. Register forward hooks on the exact named encoder modules.
    3. Iterate over identical image batches under ``torch.inference_mode`` and
       globally average pool each captured module output.
    4. Accumulate per-layer sums and self-products plus one concatenated
       teacher-by-student cross-product matrix.
    5. Remove all hooks even if evaluation raises an exception.
    6. Apply global centering analytically from the accumulated first moments,
       then normalize every named teacher/student layer pair into its CKA score.

    Parameters
    ----------
    teacher : nn.Module
        Teacher segmentation model whose encoder representations form rows of
        the output matrix.
    student : nn.Module
        Student segmentation model whose encoder representations form columns
        of the output matrix.
    teacher_layer_names : Sequence[str]
        Ordered teacher encoder module paths whose outputs form matrix rows.
        Names may be encoder-relative (``layer3``) or prefixed with
        ``encoder.``.
    student_layer_names : Sequence[str]
        Ordered student encoder module paths whose outputs form matrix columns.
        Names may be encoder-relative (``features.13``) or prefixed with
        ``encoder.``.
    dataloader : DataLoader[Tuple[ImageTensor, MaskTensor]]
        Deterministic evaluation loader. Masks are ignored, but the template's
        dataset loader is reused so image preprocessing remains identical.
    device : torch.device
        CPU or CUDA device on which inference and matrix accumulation run.

    Returns
    -------
    Tuple[CKAMatrix, List[str], List[str], int]
        CKA matrix, teacher named-module labels, student named-module labels,
        and the number of images included in the calculation.

    Raises
    ------
    AttributeError
        If either loaded model does not expose an encoder module.
    TypeError
        If a selected module returns a non-tensor output.
    ValueError
        If layer names are invalid, a selected module is missing or executes
        ambiguously, the dataloader is empty, feature widths change across
        batches, or a selected output has zero centered variance.
    """
    teacher.eval()
    student.eval()

    # Resolve all paths relative to the encoders and validate them before the
    # expensive dataset pass so typos fail immediately.
    normalized_teacher_layers = normalize_encoder_layer_names(teacher_layer_names)
    normalized_student_layers = normalize_encoder_layer_names(student_layer_names)
    teacher_encoder = get_encoder(teacher)
    student_encoder = get_encoder(student)

    teacher_activations: ActivationStore = {}
    student_activations: ActivationStore = {}
    teacher_handles: List[RemovableHandle] = []
    student_handles: List[RemovableHandle] = []

    teacher_labels: List[str] = []
    student_labels: List[str] = []
    teacher_widths: List[int] = []
    student_widths: List[int] = []

    teacher_sums: List[Tensor] = []
    student_sums: List[Tensor] = []
    teacher_second_moments: List[Tensor] = []
    student_second_moments: List[Tensor] = []
    cross_moment: Tensor | None = None
    teacher_total_sum: Tensor | None = None
    student_total_sum: Tensor | None = None
    sample_count = 0

    try:
        # Register hooks inside the protected region so hooks already attached
        # to one model are still removed if registration for the other fails.
        teacher_handles = register_named_activation_hooks(
            teacher_encoder,
            normalized_teacher_layers,
            teacher_activations,
            "Teacher",
        )
        student_handles = register_named_activation_hooks(
            student_encoder,
            normalized_student_layers,
            student_activations,
            "Student",
        )

        with torch.inference_mode():
            for batch_index, (images, _) in enumerate(dataloader):
                # The same image rows are fed to both encoders, which is required
                # because CKA compares geometry over corresponding observations.
                images = images.to(device)
                teacher_reps, current_teacher_labels = (
                    extract_named_encoder_representations(
                        teacher_encoder,
                        images,
                        normalized_teacher_layers,
                        teacher_activations,
                        "Teacher",
                    )
                )
                student_reps, current_student_labels = (
                    extract_named_encoder_representations(
                        student_encoder,
                        images,
                        normalized_student_layers,
                        student_activations,
                        "Student",
                    )
                )

                current_teacher_widths = [int(rep.shape[1]) for rep in teacher_reps]
                current_student_widths = [int(rep.shape[1]) for rep in student_reps]

                if batch_index == 0:
                    teacher_labels = current_teacher_labels
                    student_labels = current_student_labels
                    teacher_widths = current_teacher_widths
                    student_widths = current_student_widths

                    # Accumulate first/second moments in float64 to reduce the
                    # cancellation error introduced by global centering.
                    teacher_sums = [
                        torch.zeros(width, dtype=torch.float64, device=device)
                        for width in teacher_widths
                    ]
                    student_sums = [
                        torch.zeros(width, dtype=torch.float64, device=device)
                        for width in student_widths
                    ]
                    teacher_second_moments = [
                        torch.zeros((width, width), dtype=torch.float64, device=device)
                        for width in teacher_widths
                    ]
                    student_second_moments = [
                        torch.zeros((width, width), dtype=torch.float64, device=device)
                        for width in student_widths
                    ]

                    teacher_total_width = sum(teacher_widths)
                    student_total_width = sum(student_widths)
                    cross_moment = torch.zeros(
                        (teacher_total_width, student_total_width),
                        dtype=torch.float64,
                        device=device,
                    )
                    teacher_total_sum = torch.zeros(
                        teacher_total_width, dtype=torch.float64, device=device
                    )
                    student_total_sum = torch.zeros(
                        student_total_width, dtype=torch.float64, device=device
                    )
                else:
                    # Stable widths ensure each accumulated matrix block keeps
                    # referring to the same named layer/channel across batches.
                    if current_teacher_widths != teacher_widths:
                        raise ValueError(
                            "Teacher encoder feature widths changed across batches: "
                            f"expected {teacher_widths}, got {current_teacher_widths}."
                        )
                    if current_student_widths != student_widths:
                        raise ValueError(
                            "Student encoder feature widths changed across batches: "
                            f"expected {student_widths}, got {current_student_widths}."
                        )

                if (
                    cross_moment is None
                    or teacher_total_sum is None
                    or student_total_sum is None
                ):
                    raise ValueError(
                        "CKA accumulators were not initialized from the first batch."
                    )

                # Concatenating only pooled named-layer vectors lets one GEMM
                # accumulate every teacher-layer x student-layer cross-product.
                teacher_concat = torch.cat(teacher_reps, dim=1)
                student_concat = torch.cat(student_reps, dim=1)
                batch_size = int(teacher_concat.shape[0])
                sample_count += batch_size

                # Matrix multiplications run in the model's native precision for
                # GPU efficiency; promote their results before accumulation.
                cross_moment += (teacher_concat.T @ student_concat).to(torch.float64)
                teacher_total_sum += teacher_concat.sum(dim=0).to(torch.float64)
                student_total_sum += student_concat.sum(dim=0).to(torch.float64)

                for layer_index, representation in enumerate(teacher_reps):
                    teacher_sums[layer_index] += representation.sum(dim=0).to(
                        torch.float64
                    )
                    teacher_second_moments[layer_index] += (
                        representation.T @ representation
                    ).to(torch.float64)

                for layer_index, representation in enumerate(student_reps):
                    student_sums[layer_index] += representation.sum(dim=0).to(
                        torch.float64
                    )
                    student_second_moments[layer_index] += (
                        representation.T @ representation
                    ).to(torch.float64)

                print(
                    f"  Processed batch {batch_index + 1}/{len(dataloader)} "
                    f"({sample_count} images)."
                )
    finally:
        # Forward hooks persist on modules until explicitly removed. Always
        # detach them so later evaluation/inference cannot accidentally keep
        # collecting activations or retaining large tensors.
        for handle in teacher_handles + student_handles:
            handle.remove()

    if (
        sample_count == 0
        or cross_moment is None
        or teacher_total_sum is None
        or student_total_sum is None
    ):
        raise ValueError("The CKA dataloader yielded no images.")

    # Center the concatenated teacher-student cross moment with the exact global
    # dataset means; this is equivalent to explicitly forming X_c and Y_c.
    centered_cross = cross_moment - torch.outer(
        teacher_total_sum, student_total_sum
    ) / float(sample_count)

    teacher_centered_self: List[Tensor] = []
    student_centered_self: List[Tensor] = []
    for layer_index, second_moment in enumerate(teacher_second_moments):
        centered = second_moment - torch.outer(
            teacher_sums[layer_index], teacher_sums[layer_index]
        ) / float(sample_count)
        teacher_centered_self.append(centered)

    for layer_index, second_moment in enumerate(student_second_moments):
        centered = second_moment - torch.outer(
            student_sums[layer_index], student_sums[layer_index]
        ) / float(sample_count)
        student_centered_self.append(centered)

    # Convert layer widths to slice boundaries so each heatmap cell receives the
    # correct block from the one large centered cross-product matrix.
    teacher_offsets = np.cumsum([0] + teacher_widths).tolist()
    student_offsets = np.cumsum([0] + student_widths).tolist()
    cka = np.zeros((len(teacher_widths), len(student_widths)), dtype=np.float64)

    for teacher_index in range(len(teacher_widths)):
        teacher_self_energy = torch.sum(
            teacher_centered_self[teacher_index] ** 2
        )
        if teacher_self_energy <= 0:
            raise ValueError(
                f"Teacher layer '{teacher_labels[teacher_index]}' has zero centered "
                "variance; linear CKA is undefined for that output."
            )

        teacher_start = teacher_offsets[teacher_index]
        teacher_end = teacher_offsets[teacher_index + 1]

        for student_index in range(len(student_widths)):
            student_self_energy = torch.sum(
                student_centered_self[student_index] ** 2
            )
            if student_self_energy <= 0:
                raise ValueError(
                    f"Student layer '{student_labels[student_index]}' has zero "
                    "centered variance; linear CKA is undefined for that output."
                )

            student_start = student_offsets[student_index]
            student_end = student_offsets[student_index + 1]
            cross_block = centered_cross[
                teacher_start:teacher_end,
                student_start:student_end,
            ]

            numerator = torch.sum(cross_block ** 2)
            denominator = torch.sqrt(teacher_self_energy * student_self_energy)

            # The exact statistic lies in [0, 1]. Clamp only tiny numerical
            # excursions caused by finite precision, not the underlying data.
            score = torch.clamp(numerator / denominator, min=0.0, max=1.0)
            cka[teacher_index, student_index] = float(score.item())

    return cast(CKAMatrix, cka.astype(np.float32)), teacher_labels, student_labels, sample_count


@jaxtyped(typechecker=beartype)
def save_cka_outputs(
    cka_matrix: CKAMatrix,
    teacher_labels: List[str],
    student_labels: List[str],
    output_dir: str,
) -> Tuple[str, str]:
    """Save the CKA matrix as both CSV data and an annotated heatmap.

    The CSV preserves exact numeric results for later statistical analysis,
    while both CSV axes and heatmap ticks preserve the exact named module whose
    output generated each representation. This makes every CKA cell auditable
    without consulting an external T1/S1 stage mapping.

    Steps
    -----
    1. Create ``output_dir`` if it does not already exist.
    2. Write the matrix to ``cka_matrix.csv`` with named-layer labels on both
       axes.
    3. Render a fixed-range ``[0, 1]`` heatmap and annotate every cell.
    4. Save the figure to ``cka_heatmap.png``.

    Parameters
    ----------
    cka_matrix : CKAMatrix
        Two-dimensional CKA array with teacher module outputs on rows and
        student module outputs on columns.
    teacher_labels : List[str]
        Row labels naming the teacher module outputs being compared.
    student_labels : List[str]
        Column labels naming the student module outputs being compared.
    output_dir : str
        Directory where CSV and PNG outputs are written.

    Returns
    -------
    Tuple[str, str]
        Paths to the saved CSV file and heatmap PNG, respectively.

    Raises
    ------
    ValueError
        If the matrix shape does not match the supplied named-layer labels.
    """
    expected_shape = (len(teacher_labels), len(student_labels))
    if tuple(cka_matrix.shape) != expected_shape:
        raise ValueError(
            f"CKA matrix shape {tuple(cka_matrix.shape)} does not match labels "
            f"{expected_shape}."
        )

    # Persist the raw scores separately from the figure so downstream analysis
    # never has to recover values from image annotations.
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cka_matrix.csv")
    heatmap_path = os.path.join(output_dir, "cka_heatmap.png")
    cka_df = pd.DataFrame(
        cka_matrix,
        index=[label.replace("\n", " | ") for label in teacher_labels],
        columns=[label.replace("\n", " | ") for label in student_labels],
    )
    cka_df.to_csv(csv_path, index_label="teacher_layer_output")

    # Keep a fixed [0, 1] scale so multiple experiments (for example KD versus
    # a non-distilled baseline) remain visually comparable without re-scaling.
    figure_width = max(8.0, 1.6 * len(student_labels) + 3.0)
    figure_height = max(6.0, 1.2 * len(teacher_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(cka_matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(len(student_labels)))
    ax.set_yticks(np.arange(len(teacher_labels)))
    ax.set_xticklabels(student_labels, rotation=45, ha="right")
    ax.set_yticklabels(teacher_labels)
    ax.set_xlabel("Student encoder module outputs")
    ax.set_ylabel("Teacher encoder module outputs")
    ax.set_title("Teacher-Student Linear CKA (Named Encoder Outputs)")

    for teacher_index in range(cka_matrix.shape[0]):
        for student_index in range(cka_matrix.shape[1]):
            value = float(cka_matrix[teacher_index, student_index])
            text_color = "white" if value < 0.5 else "black"
            ax.text(
                student_index,
                teacher_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Linear CKA")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return csv_path, heatmap_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for teacher-student CKA evaluation.

    The teacher and student checkpoints are mandatory. Layer-selection options
    default to the ResNet-18 and MobileNetV2 endpoints used by this experiment,
    while runtime options control DataLoader throughput, deterministic subset
    size, and output location.

    Steps
    -----
    1. Require separate teacher and student checkpoint paths.
    2. Add explicit teacher/student encoder module selections with architecture-
       appropriate defaults.
    3. Add optional batch-size, maximum-sample, and output-directory controls.
    4. Parse and return the command-line namespace.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments containing checkpoint paths, selected named encoder
        layers, DataLoader settings, optional sample cap, and output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare explicitly named teacher and student encoder module "
            "outputs with linear Centered Kernel Alignment (CKA)."
        )
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Path to the trained teacher PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="Path to the trained student PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--teacher-layers",
        type=str,
        nargs="+",
        default=list(DEFAULT_TEACHER_LAYERS),
        help=(
            "Teacher encoder module outputs to compare, in matrix-row order. "
            "Names are relative to model.encoder; an 'encoder.' prefix is also "
            "accepted. Defaults to ResNet-18 relu/layer1/layer2/layer3/layer4."
        ),
    )
    parser.add_argument(
        "--student-layers",
        type=str,
        nargs="+",
        default=list(DEFAULT_STUDENT_LAYERS),
        help=(
            "Student encoder module outputs to compare, in matrix-column order. "
            "Names are relative to model.encoder; an 'encoder.' prefix is also "
            "accepted. Defaults to MobileNetV2 feature endpoints 1/3/6/13/18."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Configuration.BATCH_SIZE,
        help="DataLoader batch size. Defaults to the performance-evaluation configuration.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional maximum number of test images used for CKA. "
            "By default the complete held-out test split is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model/cka",
        help="Directory for cka_matrix.csv and cka_heatmap.png.",
    )
    return parser.parse_args()


def main() -> None:
    """Load teacher/student checkpoints and export their encoder CKA matrix.

    The pipeline deliberately reuses the test split and inference resize from
    ``evaluate_performance.py`` so CKA is measured on the same held-out images
    used for predictive evaluation. Explicit named-module hooks make every CKA
    row/column traceable to an exact encoder output, while global average
    pooling plus streaming sufficient statistics keeps the analysis practical
    for ResNet-18 and MobileNetV2.

    Steps
    -----
    1. Parse teacher/student checkpoint paths and runtime options.
    2. Load both serialized models on the configured evaluation device.
    3. Report the exact named encoder outputs selected for teacher and student.
    4. Recreate the deterministic BDD100k held-out test split and preprocessing.
    5. Optionally take the first ``max_samples`` rows for a reproducible subset.
    6. Compute all selected named-layer pair CKA scores in one dataset pass.
    7. Save the numeric matrix and annotated heatmap and print their locations.

    Raises
    ------
    ValueError
        If batch size or maximum sample count is not positive, or if no test
        samples remain after applying the requested cap.
    """
    args = parse_arguments()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero when provided.")

    print("\n" + "=" * 68)
    print("CENTERED KERNEL ALIGNMENT (CKA) PIPELINE")
    print("=" * 68)
    print(f"Target execution device: {Configuration.DEVICE}")

    # Load full nn.Module checkpoints exactly as the performance evaluation does
    # so no architecture reconstruction or state-dict assumptions are introduced.
    print(f"Loading teacher checkpoint: {args.teacher}")
    teacher = load_trained_model(args.teacher, Configuration.DEVICE)
    print(f"Loading student checkpoint: {args.student}")
    student = load_trained_model(args.student, Configuration.DEVICE)

    # Normalize now so the log shows the canonical paths that will appear in
    # the CKA output. The computation validates that these paths actually exist
    # in each loaded encoder before registering hooks.
    teacher_layer_names = normalize_encoder_layer_names(args.teacher_layers)
    student_layer_names = normalize_encoder_layer_names(args.student_layers)
    print("Teacher encoder outputs selected for CKA:")
    for layer_name in teacher_layer_names:
        print(f"  - encoder.{layer_name}")
    print("Student encoder outputs selected for CKA:")
    for layer_name in student_layer_names:
        print(f"  - encoder.{layer_name}")

    # Recreate the identical held-out split used by the template. Restricting
    # with ``head`` is deterministic and therefore suitable for repeatable pilot
    # CKA runs before using the complete test partition.
    print("Loading BDD100k test split...")
    _, _, test_df = load_dataset_from_files()
    if args.max_samples is not None:
        test_df = test_df.head(min(args.max_samples, len(test_df))).reset_index(drop=True)
    if test_df.empty:
        raise ValueError("The selected BDD100k test split contains no samples.")
    print(f"Using {len(test_df)} test samples for CKA.")

    inference_transforms = A.Compose([
        A.Resize(height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH),
        ToTensorV2(),
    ])
    test_ds = BDDSegmentationDataset(test_df, transform=inference_transforms)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=False,  # identical row ordering is mandatory for teacher/student CKA
        num_workers=Configuration.NUM_WORKERS,
    )

    # Print the teacher architecture so selected hook paths can be checked
    # against the model summary in the same execution log.
    print(
        summary(
            model=unwrap_model(teacher),
            input_size=(
                args.batch_size,
                Configuration.CHANNELS,
                Configuration.IMAGE_HEIGHT,
                Configuration.IMAGE_WIDTH,
            ),
            col_names=["output_size", "num_params", "trainable"],
            col_width=30,
            row_settings=["var_names"],
            depth=5,
            verbose=0,
        )
    )

    # Print the student architecture for the same traceability reason.
    print(
        summary(
            model=unwrap_model(student),
            input_size=(
                args.batch_size,
                Configuration.CHANNELS,
                Configuration.IMAGE_HEIGHT,
                Configuration.IMAGE_WIDTH,
            ),
            col_names=["output_size", "num_params", "trainable"],
            col_width=30,
            row_settings=["var_names"],
            depth=5,
            verbose=0,
        )
    )

    print("Computing named-layer linear CKA...")
    cka_matrix, teacher_labels, student_labels, sample_count = compute_linear_cka_matrix(
        teacher=teacher,
        student=student,
        teacher_layer_names=teacher_layer_names,
        student_layer_names=student_layer_names,
        dataloader=test_loader,
        device=Configuration.DEVICE,
    )

    # Print the matrix before exporting so headless/remote runs still expose the
    # quantitative result directly in their logs.
    result_df = pd.DataFrame(
        cka_matrix,
        index=[label.replace("\n", " | ") for label in teacher_labels],
        columns=[label.replace("\n", " | ") for label in student_labels],
    )
    print(f"\nLinear CKA over {sample_count} images:")
    print(result_df.round(4).to_string())

    csv_path, heatmap_path = save_cka_outputs(
        cka_matrix=cka_matrix,
        teacher_labels=teacher_labels,
        student_labels=student_labels,
        output_dir=args.output_dir,
    )
    print(f"\nSaved CKA matrix : {csv_path}")
    print(f"Saved CKA heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
