"""Centered Kernel Alignment evaluation for teacher and student U-Net checkpoints.

This script mirrors the BDD100k loading and checkpoint conventions used by
``evaluate_performance.py`` and compares representative internal outputs across
the complete teacher and student segmentation pathways with linear Centered
Kernel Alignment (CKA).

The default comparison includes five encoder endpoints, all five U-Net decoder
block outputs, and the final segmentation-head output. Each selected
``(B, C, H, W)`` output is globally average pooled to ``(B, C)`` inside its
forward hook. Pooling inside the hook keeps high-resolution decoder activations
from being retained after their producing module has finished. Dataset-level CKA
statistics are accumulated batch-by-batch, so the implementation does not keep
the complete activation dataset or any ``N x N`` Gram matrix in memory.
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
# feature axis permits teacher/student layers to have different channel widths,
# while the named batch axis verifies that rows correspond to identical images.
PooledRepresentation = Float[Tensor, "b _feature"]
CKAMatrix = Float[np.ndarray, "teacher_layer student_layer"]
FeatureMapTensor = Float[Tensor, "b c h w"]
ActivationShape = Tuple[int, int, int]
ActivationRecord = Tuple[PooledRepresentation, ActivationShape]
ActivationStore = Dict[str, List[ActivationRecord]]
ForwardHook = Callable[[nn.Module, Tuple[object, ...], object], None]


# These defaults sample the complete U-Net computation path at meaningful stage
# boundaries instead of restricting CKA to the encoders. ``decoder.center`` is
# intentionally omitted because it is Identity for these checkpoints and would
# duplicate the deepest encoder representation. Individual convolutions inside
# each decoder block are also omitted because the block output is the natural
# stage representation passed to the following decoder block.
DEFAULT_TEACHER_LAYERS: Tuple[str, ...] = (
    "encoder.relu",
    "encoder.layer1",
    "encoder.layer2",
    "encoder.layer3",
    "encoder.layer4",
    "decoder.blocks.0",
    "decoder.blocks.1",
    "decoder.blocks.2",
    "decoder.blocks.3",
    "decoder.blocks.4",
    "segmentation_head",
)
DEFAULT_STUDENT_LAYERS: Tuple[str, ...] = (
    "encoder.features.1",
    "encoder.features.3",
    "encoder.features.6",
    "encoder.features.13",
    "encoder.features.18",
    "decoder.blocks.0",
    "decoder.blocks.1",
    "decoder.blocks.2",
    "decoder.blocks.3",
    "decoder.blocks.4",
    "segmentation_head",
)


def parse_checkpoint_spec(specification: str) -> Tuple[str, str]:
    """Parse one required named checkpoint specification.

    CKA output directories use checkpoint names directly, so every checkpoint
    must provide an explicit, non-empty ``NAME=PATH`` pair rather than relying
    on a path-derived implicit name.

    Steps
    -----
    1. Require an equals sign separating the checkpoint name and path.
    2. Split ``NAME=PATH`` at the first equals sign.
    3. Strip surrounding whitespace from both values.
    4. Reject empty names or paths before checkpoint loading begins.

    Parameters
    ----------
    specification : str
        Checkpoint specification in mandatory ``NAME=PATH`` form.

    Returns
    -------
    Tuple[str, str]
        Human-readable checkpoint name and checkpoint path.

    Raises
    ------
    ValueError
        If ``specification`` does not use ``NAME=PATH`` form or either value is
        empty.
    """
    if "=" not in specification:
        raise ValueError(
            "Checkpoint values must use NAME=PATH with an explicit name."
        )

    # Split only once so a filesystem path containing '=' remains valid while
    # the user-provided prefix continues to define the checkpoint's output name.
    model_name, checkpoint_path = specification.split("=", maxsplit=1)
    model_name = model_name.strip()
    checkpoint_path = checkpoint_path.strip()
    if not model_name or not checkpoint_path:
        raise ValueError(
            "Checkpoint values must use NAME=PATH with non-empty values."
        )
    return model_name, checkpoint_path


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when evaluation uses ``nn.DataParallel``.

    CKA hooks must resolve names against the actual segmentation model rather
    than against the outer ``DataParallel`` wrapper. This helper normalizes both
    checkpoint forms before module lookup.

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
        Underlying segmentation model used for named-module lookup and forward
        execution.
    """
    # ``DataParallel`` stores the user model under ``module``. Unwrapping keeps
    # CLI paths such as ``decoder.blocks.0`` independent of checkpoint wrapping.
    return model.module if isinstance(model, nn.DataParallel) else model


def normalize_layer_names(layer_names: Sequence[str]) -> List[str]:
    """Normalize requested CKA module paths relative to the full model.

    Full paths such as ``encoder.layer3`` and ``decoder.blocks.1`` are preserved.
    For backward compatibility with the earlier encoder-only script, an input
    such as ``layer3`` or ``features.13`` is interpreted as encoder-relative and
    receives the ``encoder.`` prefix.

    Steps
    -----
    1. Strip whitespace from each requested path.
    2. Preserve paths rooted at ``encoder``, ``decoder``, or
       ``segmentation_head``.
    3. Prefix any legacy encoder-relative path with ``encoder.``.
    4. Reject empty or duplicate paths because ambiguous matrix labels would
       make the exported comparison difficult to audit.

    Parameters
    ----------
    layer_names : Sequence[str]
        Requested full-model module paths or legacy encoder-relative paths.

    Returns
    -------
    List[str]
        Ordered full-model module paths suitable for ``model.named_modules()``.

    Raises
    ------
    ValueError
        If no layer names are supplied, a path is empty, or a path is repeated.
    """
    normalized: List[str] = []
    for layer_name in layer_names:
        # Preserve explicit full-model paths. Bare names from the prior CLI are
        # treated as encoder-relative so existing commands keep working.
        name = layer_name.strip()
        if not name:
            raise ValueError("CKA layer names must not be empty.")
        if not (
            name.startswith("encoder.")
            or name.startswith("decoder.")
            or name == "segmentation_head"
            or name.startswith("segmentation_head.")
        ):
            name = f"encoder.{name}"
        normalized.append(name)

    if not normalized:
        raise ValueError("At least one model layer must be selected for CKA.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"CKA layer names must be unique, got {normalized}.")
    return normalized


@jaxtyped(typechecker=beartype)
def global_average_pool(feature_map: FeatureMapTensor) -> PooledRepresentation:
    """Collapse a convolutional feature map to one channel vector per image.

    Global average pooling makes stage-level CKA practical across encoder,
    decoder, and output resolutions. It preserves one aggregate response per
    channel while intentionally discarding spatial location, leaving spatial
    explanation to complementary analyses such as Grad-CAM or mask comparison.

    Steps
    -----
    1. Average every channel over its complete spatial extent.
    2. Flatten the two singleton spatial dimensions to obtain ``(B, C)``.

    Parameters
    ----------
    feature_map : Float[Tensor, "b c h w"]
        Selected module activation with shape ``(B, C, H, W)``.

    Returns
    -------
    PooledRepresentation
        Per-image representation matrix with shape ``(B, C)``.
    """
    # Adaptive pooling works for every resolution in the encoder, decoder, and
    # segmentation head without assuming teacher/student spatial equality.
    return cast(
        PooledRepresentation,
        F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(start_dim=1),
    )


def make_activation_hook(
    layer_name: str,
    activation_store: ActivationStore,
    model_label: str,
) -> ForwardHook:
    """Create a forward hook that stores one pooled named-module output.

    The hook performs global average pooling immediately rather than retaining
    the complete feature map until the end of the U-Net forward pass. This is
    especially important for decoder outputs such as ``(B, 16, 360, 640)``.
    Shape metadata is retained so exported labels still identify the original
    tensor that produced each pooled representation.

    Steps
    -----
    1. Receive the selected module's forward output.
    2. Require a four-dimensional tensor suitable for convolutional CKA.
    3. Record the original channel and spatial dimensions.
    4. Global-average-pool the tensor and append only the compact representation
       plus its original shape metadata.

    Parameters
    ----------
    layer_name : str
        Full module path being observed.
    activation_store : ActivationStore
        Mutable mapping from module path to pooled outputs captured in the
        current model forward pass.
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
    ValueError
        If the selected module output is not shaped ``(B, C, H, W)``.
    """

    def capture_output(
        _module: nn.Module,
        _inputs: Tuple[object, ...],
        output: object,
    ) -> None:
        # A tuple/dict output would need an explicit policy for choosing the
        # representation, so fail rather than selecting an element silently.
        if not isinstance(output, Tensor):
            raise TypeError(
                f"{model_label} layer '{layer_name}' returned {type(output)}, "
                "but CKA requires a Tensor output."
            )
        if output.ndim != 4:
            raise ValueError(
                f"{model_label} layer '{layer_name}' has output shape "
                f"{tuple(output.shape)}; CKA expects (B, C, H, W)."
            )

        # Pool inside the hook so full-resolution decoder activations can be
        # released by PyTorch as the forward pass progresses.
        _, channels, height, width = output.shape
        pooled = global_average_pool(cast(FeatureMapTensor, output))
        activation_store[layer_name].append(
            (pooled, (int(channels), int(height), int(width)))
        )

    return capture_output


def register_named_activation_hooks(
    model: nn.Module,
    layer_names: Sequence[str],
    activation_store: ActivationStore,
    model_label: str,
) -> List[RemovableHandle]:
    """Register forward hooks on explicitly selected full-model modules.

    Paths are resolved against ``model.named_modules()`` before evaluation. This
    makes every CKA matrix coordinate traceable to an actual module in the
    loaded checkpoint and allows encoder, decoder, and segmentation-head outputs
    to participate in the same analysis.

    Steps
    -----
    1. Build a lookup from ``model.named_modules()``.
    2. Validate that every requested module path exists.
    3. Initialize one activation list per selected path.
    4. Register a forward hook on each selected module in requested order.

    Parameters
    ----------
    model : nn.Module
        Unwrapped teacher or student segmentation model.
    layer_names : Sequence[str]
        Ordered full-model module paths to capture.
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
        If a requested module path does not exist in the model.
    """
    named_modules = dict(model.named_modules())
    missing_layers = [name for name in layer_names if name not in named_modules]
    if missing_layers:
        raise ValueError(
            f"{model_label} does not contain requested CKA module(s): "
            f"{missing_layers}. Use names from model.named_modules()."
        )

    handles: List[RemovableHandle] = []
    for layer_name in layer_names:
        # Pre-create the list so the hook can append without changing mapping
        # keys while the model is executing.
        activation_store[layer_name] = []
        hook = make_activation_hook(layer_name, activation_store, model_label)
        handles.append(named_modules[layer_name].register_forward_hook(hook))
    return handles


@jaxtyped(typechecker=beartype)
def extract_named_representations(
    model: nn.Module,
    x: BatchImage,
    layer_names: Sequence[str],
    activation_store: ActivationStore,
    model_label: str,
) -> Tuple[List[PooledRepresentation], List[str]]:
    """Run the full model and collect pooled outputs from named modules.

    Forward hooks define the exact representations used by CKA. The entire U-Net
    is executed once per batch so selected encoder, decoder, and segmentation
    head modules all fire. Because pooling occurs inside each hook, only compact
    ``(B, C)`` representations survive after their module executes.

    Steps
    -----
    1. Reset the hook store for every selected module.
    2. Run one full model forward pass.
    3. Require every selected module to execute exactly once, avoiding ambiguous
       outputs from reused modules.
    4. Collect the pooled representation and construct a label containing the
       exact module path plus its original channel/spatial dimensions.
    5. Clear hook stores so no batch tensors remain referenced unnecessarily.

    Parameters
    ----------
    model : nn.Module
        Unwrapped segmentation model whose selected modules have hooks.
    x : BatchImage
        Input image batch with shape ``(B, 3, H, W)``.
    layer_names : Sequence[str]
        Ordered full-model module paths whose outputs are compared.
    activation_store : ActivationStore
        Hook-populated mapping from module path to pooled outputs and shapes.
    model_label : str
        Human-readable model label used in validation errors.

    Returns
    -------
    Tuple[List[PooledRepresentation], List[str]]
        Pooled named-module representations and matching traceable labels.

    Raises
    ------
    ValueError
        If a selected module is not executed exactly once in the model forward.
    """
    # Reset prior batch records so execution-count validation describes only the
    # current forward pass.
    for layer_name in layer_names:
        activation_store[layer_name] = []

    # A complete forward is required because decoder and segmentation-head
    # modules do not execute when only ``model.encoder`` is called.
    model(x)

    representations: List[PooledRepresentation] = []
    labels: List[str] = []
    try:
        for layer_name in layer_names:
            captured_outputs = activation_store[layer_name]
            if len(captured_outputs) != 1:
                raise ValueError(
                    f"{model_label} layer '{layer_name}' executed "
                    f"{len(captured_outputs)} times in one model forward; CKA "
                    "requires an unambiguous single output per selected layer."
                )

            pooled, (channels, height, width) = captured_outputs[0]
            representations.append(cast(PooledRepresentation, pooled))
            labels.append(
                f"{layer_name}\nC={channels}, {height}x{width}"
            )
    finally:
        # Remove references to pooled batch tensors after copying them into the
        # returned list. Full-resolution activations were never retained here.
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
    """Compute full-model named-layer linear CKA with streaming statistics.

    For centered representation matrices ``X`` and ``Y``, linear CKA is
    ``||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F)``. The selected
    representations may come from the encoder, decoder, or segmentation head.
    Instead of retaining all image activations, this function accumulates the
    first and second moments required to center the complete dataset exactly.

    Steps
    -----
    1. Normalize and validate teacher/student full-model module paths.
    2. Register hooks on every exact selected module.
    3. Iterate over identical image batches under ``torch.inference_mode`` and
       run each complete U-Net once per batch.
    4. Accumulate per-layer sums and self-products plus one concatenated
       teacher-by-student cross-product matrix.
    5. Remove all hooks even if evaluation raises an exception.
    6. Apply dataset-wide centering analytically and normalize every selected
       teacher/student module pair into its linear CKA score.

    Parameters
    ----------
    teacher : nn.Module
        Teacher segmentation model whose selected outputs form matrix rows.
    student : nn.Module
        Student segmentation model whose selected outputs form matrix columns.
    teacher_layer_names : Sequence[str]
        Ordered teacher module paths. Bare legacy names are interpreted as
        encoder-relative paths.
    student_layer_names : Sequence[str]
        Ordered student module paths. Bare legacy names are interpreted as
        encoder-relative paths.
    dataloader : DataLoader[Tuple[ImageTensor, MaskTensor]]
        Deterministic evaluation loader. Masks are ignored, but the template's
        dataset loader is reused so image preprocessing remains identical.
    device : torch.device
        CPU or CUDA device on which inference and matrix accumulation run.

    Returns
    -------
    Tuple[CKAMatrix, List[str], List[str], int]
        CKA matrix, teacher named-module labels, student named-module labels,
        and number of images included in the calculation.

    Raises
    ------
    TypeError
        If a selected module returns a non-tensor output.
    ValueError
        If layer names are invalid, a selected module is missing or executes
        ambiguously, the dataloader is empty, feature widths change across
        batches, or a selected output has zero centered variance.
    """
    teacher.eval()
    student.eval()

    # Resolve names against unwrapped full models so decoder/head modules can be
    # selected in addition to encoder endpoints.
    normalized_teacher_layers = normalize_layer_names(teacher_layer_names)
    normalized_student_layers = normalize_layer_names(student_layer_names)
    teacher_model = unwrap_model(teacher)
    student_model = unwrap_model(student)

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
        # Hook registration is inside the protected region so partially
        # registered hooks are still removed if the second model fails lookup.
        teacher_handles = register_named_activation_hooks(
            teacher_model,
            normalized_teacher_layers,
            teacher_activations,
            "Teacher",
        )
        student_handles = register_named_activation_hooks(
            student_model,
            normalized_student_layers,
            student_activations,
            "Student",
        )

        with torch.inference_mode():
            for batch_index, (images, _) in enumerate(dataloader):
                # Both models receive identical image rows because CKA compares
                # representational geometry over corresponding observations.
                images = images.to(device)
                teacher_reps, current_teacher_labels = extract_named_representations(
                    teacher_model,
                    images,
                    normalized_teacher_layers,
                    teacher_activations,
                    "Teacher",
                )
                student_reps, current_student_labels = extract_named_representations(
                    student_model,
                    images,
                    normalized_student_layers,
                    student_activations,
                    "Student",
                )

                current_teacher_widths = [int(rep.shape[1]) for rep in teacher_reps]
                current_student_widths = [int(rep.shape[1]) for rep in student_reps]

                if batch_index == 0:
                    teacher_labels = current_teacher_labels
                    student_labels = current_student_labels
                    teacher_widths = current_teacher_widths
                    student_widths = current_student_widths

                    # Accumulate in float64 to reduce cancellation error when
                    # global centering subtracts dataset first moments later.
                    teacher_sums = [
                        torch.zeros(width, dtype=torch.float64, device=device)
                        for width in teacher_widths
                    ]
                    student_sums = [
                        torch.zeros(width, dtype=torch.float64, device=device)
                        for width in student_widths
                    ]
                    teacher_second_moments = [
                        torch.zeros(
                            (width, width),
                            dtype=torch.float64,
                            device=device,
                        )
                        for width in teacher_widths
                    ]
                    student_second_moments = [
                        torch.zeros(
                            (width, width),
                            dtype=torch.float64,
                            device=device,
                        )
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
                        teacher_total_width,
                        dtype=torch.float64,
                        device=device,
                    )
                    student_total_sum = torch.zeros(
                        student_total_width,
                        dtype=torch.float64,
                        device=device,
                    )
                else:
                    # Stable widths guarantee that each block of the accumulated
                    # matrices keeps referring to the same named module/channel.
                    if current_teacher_widths != teacher_widths:
                        raise ValueError(
                            "Teacher feature widths changed across batches: "
                            f"expected {teacher_widths}, got "
                            f"{current_teacher_widths}."
                        )
                    if current_student_widths != student_widths:
                        raise ValueError(
                            "Student feature widths changed across batches: "
                            f"expected {student_widths}, got "
                            f"{current_student_widths}."
                        )

                if (
                    cross_moment is None
                    or teacher_total_sum is None
                    or student_total_sum is None
                ):
                    raise ValueError(
                        "CKA accumulators were not initialized from the first batch."
                    )

                # Concatenating only already-pooled outputs lets one GEMM
                # accumulate every teacher-layer x student-layer cross-product.
                teacher_concat = torch.cat(teacher_reps, dim=1)
                student_concat = torch.cat(student_reps, dim=1)
                batch_size = int(teacher_concat.shape[0])
                sample_count += batch_size

                # GEMMs use the model's native precision for throughput, then
                # promote results before adding to dataset-level accumulators.
                cross_moment += (teacher_concat.T @ student_concat).to(
                    torch.float64
                )
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
        # Hooks persist until removed. Always detach them so later inference does
        # not accidentally keep capturing activations after CKA finishes/fails.
        for handle in teacher_handles + student_handles:
            handle.remove()

    if (
        sample_count == 0
        or cross_moment is None
        or teacher_total_sum is None
        or student_total_sum is None
    ):
        raise ValueError("The CKA dataloader yielded no images.")

    # Center the concatenated teacher-student moment using exact global dataset
    # means; this is equivalent to explicitly constructing centered X and Y.
    centered_cross = cross_moment - torch.outer(
        teacher_total_sum,
        student_total_sum,
    ) / float(sample_count)

    teacher_centered_self: List[Tensor] = []
    student_centered_self: List[Tensor] = []
    for layer_index, second_moment in enumerate(teacher_second_moments):
        centered = second_moment - torch.outer(
            teacher_sums[layer_index],
            teacher_sums[layer_index],
        ) / float(sample_count)
        teacher_centered_self.append(centered)

    for layer_index, second_moment in enumerate(student_second_moments):
        centered = second_moment - torch.outer(
            student_sums[layer_index],
            student_sums[layer_index],
        ) / float(sample_count)
        student_centered_self.append(centered)

    # Slice boundaries map each named output back out of the concatenated
    # cross-product matrix before computing its independently normalized CKA.
    teacher_offsets = np.cumsum([0] + teacher_widths).tolist()
    student_offsets = np.cumsum([0] + student_widths).tolist()
    cka = np.zeros((len(teacher_widths), len(student_widths)), dtype=np.float64)

    for teacher_index in range(len(teacher_widths)):
        teacher_self_energy = torch.sum(
            teacher_centered_self[teacher_index] ** 2
        )
        if teacher_self_energy <= 0:
            raise ValueError(
                f"Teacher layer '{teacher_labels[teacher_index]}' has zero "
                "centered variance; linear CKA is undefined for that output."
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
            denominator = torch.sqrt(
                teacher_self_energy * student_self_energy
            )

            # The mathematical value is in [0, 1]. Clamp only finite-precision
            # excursions instead of changing the underlying statistic.
            score = torch.clamp(
                numerator / denominator,
                min=0.0,
                max=1.0,
            )
            cka[teacher_index, student_index] = float(score.item())

    return (
        cast(CKAMatrix, cka.astype(np.float32)),
        teacher_labels,
        student_labels,
        sample_count,
    )


@jaxtyped(typechecker=beartype)
def save_cka_outputs(
    cka_matrix: CKAMatrix,
    teacher_labels: List[str],
    student_labels: List[str],
    output_dir: str,
) -> Tuple[str, str]:
    """Save the full-model CKA matrix as CSV data and an annotated heatmap.

    Both output axes preserve the exact module path and original feature-map
    shape. A fixed ``[0, 1]`` color scale keeps results comparable across KD and
    non-distilled baseline experiments.

    Steps
    -----
    1. Create ``output_dir`` if it does not exist.
    2. Write ``cka_matrix.csv`` with exact named-module labels on both axes.
    3. Render a fixed-range heatmap and annotate every cell with its CKA value.
    4. Save the figure as ``cka_heatmap.png``.

    Parameters
    ----------
    cka_matrix : CKAMatrix
        Two-dimensional CKA array with teacher module outputs on rows and
        student module outputs on columns.
    teacher_labels : List[str]
        Row labels naming teacher module outputs and their original shapes.
    student_labels : List[str]
        Column labels naming student module outputs and their original shapes.
    output_dir : str
        Directory where CSV and PNG outputs are written.

    Returns
    -------
    Tuple[str, str]
        Paths to the saved CSV file and heatmap PNG, respectively.

    Raises
    ------
    ValueError
        If the matrix shape does not match the supplied module labels.
    """
    expected_shape = (len(teacher_labels), len(student_labels))
    if tuple(cka_matrix.shape) != expected_shape:
        raise ValueError(
            f"CKA matrix shape {tuple(cka_matrix.shape)} does not match labels "
            f"{expected_shape}."
        )

    # Save exact values separately so later statistical analysis never needs to
    # recover numbers from heatmap annotations.
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cka_matrix.csv")
    heatmap_path = os.path.join(output_dir, "cka_heatmap.png")
    cka_df = pd.DataFrame(
        cka_matrix,
        index=[label.replace("\n", " | ") for label in teacher_labels],
        columns=[label.replace("\n", " | ") for label in student_labels],
    )
    cka_df.to_csv(csv_path, index_label="teacher_module_output")

    # The larger default set includes encoder, decoder, and output modules, so
    # allocate enough space for readable exact-path tick labels.
    figure_width = max(12.0, 1.35 * len(student_labels) + 4.0)
    figure_height = max(9.0, 1.0 * len(teacher_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(
        cka_matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xticks(np.arange(len(student_labels)))
    ax.set_yticks(np.arange(len(teacher_labels)))
    ax.set_xticklabels(student_labels, rotation=50, ha="right")
    ax.set_yticklabels(teacher_labels)
    ax.set_xlabel("Student module outputs")
    ax.set_ylabel("Teacher module outputs")
    ax.set_title("Teacher-Student Linear CKA (Encoder + Decoder + Output)")

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
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Linear CKA")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return csv_path, heatmap_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for full-model teacher-student CKA.

    Explicitly named teacher and student checkpoints are mandatory so each model
    is always paired with the correct architecture-specific layer defaults. Layer
    selections remain configurable for architecture-specific experiments.

    Steps
    -----
    1. Accept required ``--teacher NAME=PATH`` and ``--student NAME=PATH`` options.
    2. Add explicit teacher/student module selections with full U-Net defaults.
    3. Add batch-size and maximum-sample controls.
    4. Parse and return the command-line namespace.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing named teacher/student checkpoint
        specifications, selected named modules, DataLoader settings, and optional
        sample cap.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare named encoder, decoder, and segmentation-head outputs "
            "between teacher and student models with linear CKA."
        )
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help=(
            "Teacher checkpoint in NAME=PATH form. Its selected outputs become "
            "CKA matrix rows and use teacher-layer defaults unless overridden."
        ),
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help=(
            "Student checkpoint in NAME=PATH form. Its selected outputs become "
            "CKA matrix columns and use student-layer defaults unless overridden."
        ),
    )
    parser.add_argument(
        "--teacher-layers",
        type=str,
        nargs="+",
        default=list(DEFAULT_TEACHER_LAYERS),
        help=(
            "Teacher module outputs in matrix-row order. Defaults cover five "
            "ResNet-18 encoder endpoints, five U-Net decoder blocks, and the "
            "segmentation head. Bare names remain encoder-relative for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--student-layers",
        type=str,
        nargs="+",
        default=list(DEFAULT_STUDENT_LAYERS),
        help=(
            "Student module outputs in matrix-column order. Defaults cover "
            "five MobileNetV2 encoder endpoints, five U-Net decoder blocks, "
            "and the segmentation head. Bare names remain encoder-relative "
            "for backward compatibility."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Configuration.BATCH_SIZE,
        help=(
            "DataLoader batch size. Defaults to the performance-evaluation "
            "configuration."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional maximum number of test images used for CKA. By default "
            "the complete held-out test split is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load named teacher/student checkpoints and export full-pathway CKA.

    The analysis reuses the same held-out BDD100k split and inference resize as
    ``evaluate_performance.py``. Named hooks make every matrix coordinate
    traceable to an exact module output, while hook-time global average pooling
    and streaming moments keep the broader encoder/decoder analysis practical.

    Steps
    -----
    1. Parse named teacher/student checkpoint specifications and runtime options.
    2. Require distinct checkpoint names for an unambiguous result directory.
    3. Load the teacher and student models on the configured evaluation device.
    4. Normalize and report all selected encoder/decoder/output module paths.
    5. Recreate the deterministic held-out BDD100k split and preprocessing.
    6. Optionally take the first ``max_samples`` rows for a reproducible pilot.
    7. Print both architectures for module-path traceability.
    8. Compute all selected teacher-student output-pair CKA scores in one pass.
    9. Save results under ``./model/cka/{teacher_name}_with_{student_name}``.

    Raises
    ------
    ValueError
        If a checkpoint specification is invalid, teacher and student names are
        identical, batch size or maximum sample count is not positive, no test
        samples remain after applying the cap, or CKA module validation fails.
    """
    args = parse_arguments()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero when provided.")

    # Parse each role explicitly before model loading. Keeping teacher and student
    # separate prevents an argument-order mistake from applying architecture-
    # specific default layers to the wrong checkpoint.
    teacher_name, teacher_path = parse_checkpoint_spec(args.teacher)
    student_name, student_path = parse_checkpoint_spec(args.student)
    if teacher_name == student_name:
        raise ValueError("--teacher and --student must use distinct model names.")

    # Keep all pairwise CKA artifacts under a name-derived subdirectory so
    # repeated experiments do not require manual result-folder renaming.
    output_dir = os.path.join(
        "./model/cka",
        f"{teacher_name}_with_{student_name}",
    )

    print("\n" + "=" * 68)
    print("CENTERED KERNEL ALIGNMENT (CKA) PIPELINE")
    print("=" * 68)
    print(f"Target execution device: {Configuration.DEVICE}")

    # Load complete serialized modules exactly as performance evaluation does so
    # no architecture reconstruction or state-dict assumptions are introduced.
    print(f"Loading teacher checkpoint '{teacher_name}': {teacher_path}")
    teacher = load_trained_model(teacher_path, Configuration.DEVICE)
    print(f"Loading student checkpoint '{student_name}': {student_path}")
    student = load_trained_model(student_path, Configuration.DEVICE)
    print(f"CKA output directory: {output_dir}")

    # Normalize before the expensive dataset pass so the execution log records
    # the exact paths that will label matrix rows/columns.
    teacher_layer_names = normalize_layer_names(args.teacher_layers)
    student_layer_names = normalize_layer_names(args.student_layers)
    print("Teacher module outputs selected for CKA:")
    for layer_name in teacher_layer_names:
        print(f"  - {layer_name}")
    print("Student module outputs selected for CKA:")
    for layer_name in student_layer_names:
        print(f"  - {layer_name}")

    # Recreate the identical held-out split used by performance evaluation.
    # ``head`` makes pilot subsets deterministic before a complete CKA run.
    print("Loading BDD100k test split...")
    _, _, test_df = load_dataset_from_files()
    if args.max_samples is not None:
        test_df = test_df.head(
            min(args.max_samples, len(test_df))
        ).reset_index(drop=True)
    if test_df.empty:
        raise ValueError("The selected BDD100k test split contains no samples.")
    print(f"Using {len(test_df)} test samples for CKA.")

    inference_transforms = A.Compose([
        A.Resize(
            height=Configuration.IMAGE_HEIGHT,
            width=Configuration.IMAGE_WIDTH,
        ),
        ToTensorV2(),
    ])
    test_ds = BDDSegmentationDataset(test_df, transform=inference_transforms)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=False,  # teacher/student rows must correspond exactly for CKA
        num_workers=Configuration.NUM_WORKERS,
    )

    # Print architectures so every selected hook path can be cross-checked in
    # the same execution log as the CKA result.
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

    print("Computing full-model named-layer linear CKA...")
    cka_matrix, teacher_labels, student_labels, sample_count = (
        compute_linear_cka_matrix(
            teacher=teacher,
            student=student,
            teacher_layer_names=teacher_layer_names,
            student_layer_names=student_layer_names,
            dataloader=test_loader,
            device=Configuration.DEVICE,
        )
    )

    # Print exact values before exporting so remote/headless runs still expose
    # the complete quantitative result in logs.
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
        output_dir=output_dir,
    )
    print(f"\nSaved CKA matrix : {csv_path}")
    print(f"Saved CKA heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
