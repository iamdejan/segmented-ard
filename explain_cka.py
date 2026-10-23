"""Centered Kernel Alignment evaluation for teacher and student encoders.

This script mirrors the BDD100k loading and checkpoint conventions used by
``evaluate_performance.py`` and compares the internal encoder representations of
a teacher model and a student model with linear Centered Kernel Alignment (CKA).

Each encoder stage is globally average pooled from ``(B, C, H, W)`` to
``(B, C)`` before comparison. The implementation accumulates sufficient
statistics batch-by-batch, so it does not keep the complete activation dataset
or any ``N x N`` Gram matrix in memory.
"""

import argparse
import os
from typing import List, Sequence, Tuple, cast

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

from evaluate_performance import (
    BDDSegmentationDataset,
    BatchImage,
    Configuration,
    ImageTensor,
    MaskTensor,
    load_dataset_from_files,
    load_trained_model,
)


# CKA operates on one vector per image and stage. The anonymous feature axis
# allows teacher and student stages to have different channel widths while the
# named batch axis still verifies that rows correspond to the same images.
PooledRepresentation = Float[Tensor, "b _feature"]
CKAMatrix = Float[np.ndarray, "teacher_stage student_stage"]
FeatureMapTensor = Float[Tensor, "b c h w"]


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


@jaxtyped(typechecker=beartype)
def global_average_pool(feature_map: FeatureMapTensor) -> PooledRepresentation:
    """Collapse a convolutional feature map to one channel vector per image.

    CKA should compare compact stage representations rather than flattened
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
        Encoder-stage activation tensor with shape ``(B, C, H, W)``.

    Returns
    -------
    PooledRepresentation
        Per-image representation matrix with shape ``(B, C)``.
    """
    # Adaptive pooling avoids making assumptions about the spatial resolution
    # of corresponding ResNet-18 and MobileNetV2 stages.
    return cast(
        PooledRepresentation,
        F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(start_dim=1),
    )


@jaxtyped(typechecker=beartype)
def extract_encoder_representations(
    model: nn.Module,
    x: BatchImage,
    label_prefix: str,
) -> Tuple[List[PooledRepresentation], List[str]]:
    """Extract globally pooled multi-scale encoder-stage representations.

    ``segmentation_models_pytorch`` encoders return a sequence of feature maps
    at progressively deeper scales. Some encoders include the unmodified input
    tensor as the first sequence element; that tensor is excluded because it is
    not a learned representation. Every remaining 4-D feature map is globally
    average pooled so stages with different channel widths can still be
    compared by CKA.

    Steps
    -----
    1. Unwrap ``nn.DataParallel`` if necessary and obtain ``model.encoder``.
    2. Run only the encoder because decoder logits are not needed for CKA.
    3. Drop the first returned feature only when it is exactly input-shaped.
    4. Global-average-pool every learned stage to ``(B, C)``.
    5. Build labels containing stage number, channel count, and spatial size.

    Parameters
    ----------
    model : nn.Module
        Loaded segmentation model containing an ``encoder`` attribute.
    x : BatchImage
        Input image batch with shape ``(B, 3, H, W)``.
    label_prefix : str
        Short model label such as ``"T"`` or ``"S"`` used on the heatmap.

    Returns
    -------
    Tuple[List[PooledRepresentation], List[str]]
        Pooled stage representations and matching human-readable labels.

    Raises
    ------
    AttributeError
        If the loaded model does not expose an ``nn.Module`` encoder.
    TypeError
        If the encoder output is not a sequence of tensors.
    ValueError
        If no learned 4-D feature maps are available or a returned feature does
        not use the expected ``(B, C, H, W)`` layout.
    """
    base_model = unwrap_model(model)
    encoder = getattr(base_model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise AttributeError(
            f"{base_model.__class__.__name__} does not expose an nn.Module "
            "'encoder' attribute required for stage-wise CKA."
        )

    # SMP encoders return all requested depth stages in one forward call. Using
    # this public encoder interface is more stable than hard-coding ResNet or
    # MobileNet internal block names.
    raw_features = encoder(x)
    if not isinstance(raw_features, Sequence):
        raise TypeError(
            "Expected model.encoder(x) to return a sequence of feature maps, "
            f"but received {type(raw_features)}."
        )

    features = list(cast(Sequence[Tensor], raw_features))
    if not features:
        raise ValueError("The encoder returned no feature maps for CKA analysis.")

    # SMP commonly places the unchanged RGB input at index zero. Detect it from
    # shape rather than assuming every encoder follows that convention.
    first_is_raw_input = tuple(features[0].shape) == tuple(x.shape)
    learned_features = features[1:] if first_is_raw_input else features
    if not learned_features:
        raise ValueError(
            "The encoder returned no learned feature maps after filtering the raw input."
        )

    representations: List[PooledRepresentation] = []
    labels: List[str] = []
    for stage_index, feature in enumerate(learned_features, start=1):
        if feature.ndim != 4:
            raise ValueError(
                f"Encoder stage {stage_index} has shape {tuple(feature.shape)}; "
                "CKA expects convolutional features shaped (B, C, H, W)."
            )

        # Pool before accumulation so high-resolution early layers do not cause
        # the CKA computation or memory use to scale with H x W.
        pooled = global_average_pool(cast(FeatureMapTensor, feature))
        representations.append(pooled)

        _, channels, height, width = feature.shape
        labels.append(
            f"{label_prefix}{stage_index}\nC={channels}, {height}x{width}"
        )

    return representations, labels


@jaxtyped(typechecker=beartype)
def compute_linear_cka_matrix(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader[Tuple[ImageTensor, MaskTensor]],
    device: torch.device,
) -> Tuple[CKAMatrix, List[str], List[str], int]:
    """Compute an all-stage linear CKA matrix using streaming statistics.

    For centered representation matrices ``X`` and ``Y``, linear CKA is
    ``||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F)``. Instead of
    storing every pooled activation, this function accumulates the sums and
    second moments needed to construct the globally centered cross-products at
    the end. This keeps memory dependent on feature widths rather than on the
    number of evaluation images.

    Steps
    -----
    1. Put both models in evaluation mode and iterate over identical image
       batches under ``torch.inference_mode``.
    2. Extract and globally average pool every learned encoder stage.
    3. Accumulate per-stage sums and self-products plus one concatenated
       teacher-by-student cross-product matrix.
    4. Apply global centering analytically from the accumulated first moments.
    5. Slice the centered cross-product into stage pairs and normalize each
       Frobenius similarity into the final CKA score.

    Parameters
    ----------
    teacher : nn.Module
        Teacher segmentation model whose encoder representations form rows of
        the output matrix.
    student : nn.Module
        Student segmentation model whose encoder representations form columns
        of the output matrix.
    dataloader : DataLoader[Tuple[ImageTensor, MaskTensor]]
        Deterministic evaluation loader. Masks are ignored, but the template's
        dataset loader is reused so image preprocessing remains identical.
    device : torch.device
        CPU or CUDA device on which inference and matrix accumulation run.

    Returns
    -------
    Tuple[CKAMatrix, List[str], List[str], int]
        CKA matrix, teacher stage labels, student stage labels, and the number
        of images included in the calculation.

    Raises
    ------
    ValueError
        If the dataloader is empty, stage counts or widths change across
        batches, or a stage has zero centered variance and therefore cannot
        yield a defined CKA value.
    """
    teacher.eval()
    student.eval()

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

    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(dataloader):
            # The same image rows are fed to both models, which is required for
            # CKA because its geometry is defined over corresponding samples.
            images = images.to(device)
            teacher_reps, current_teacher_labels = extract_encoder_representations(
                teacher, images, "T"
            )
            student_reps, current_student_labels = extract_encoder_representations(
                student, images, "S"
            )

            current_teacher_widths = [int(rep.shape[1]) for rep in teacher_reps]
            current_student_widths = [int(rep.shape[1]) for rep in student_reps]

            if batch_index == 0:
                teacher_labels = current_teacher_labels
                student_labels = current_student_labels
                teacher_widths = current_teacher_widths
                student_widths = current_student_widths

                # Accumulate first and second moments in float64 to reduce the
                # cancellation error introduced by global centering at the end.
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
                # Stable feature widths are essential because each accumulated
                # matrix position must refer to the same stage/channel across
                # every batch.
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

            if cross_moment is None or teacher_total_sum is None or student_total_sum is None:
                raise ValueError("CKA accumulators were not initialized from the first batch.")

            # Concatenating only the already pooled stage vectors lets one GEMM
            # accumulate every teacher-stage x student-stage cross-product.
            teacher_concat = torch.cat(teacher_reps, dim=1)
            student_concat = torch.cat(student_reps, dim=1)
            batch_size = int(teacher_concat.shape[0])
            sample_count += batch_size

            # Matrix multiplications run in the model's native float precision
            # for GPU efficiency; their results are promoted before accumulation
            # so the dataset-wide sums are numerically stable.
            cross_moment += (teacher_concat.T @ student_concat).to(torch.float64)
            teacher_total_sum += teacher_concat.sum(dim=0).to(torch.float64)
            student_total_sum += student_concat.sum(dim=0).to(torch.float64)

            for stage_index, representation in enumerate(teacher_reps):
                teacher_sums[stage_index] += representation.sum(dim=0).to(torch.float64)
                teacher_second_moments[stage_index] += (
                    representation.T @ representation
                ).to(torch.float64)

            for stage_index, representation in enumerate(student_reps):
                student_sums[stage_index] += representation.sum(dim=0).to(torch.float64)
                student_second_moments[stage_index] += (
                    representation.T @ representation
                ).to(torch.float64)

            print(
                f"  Processed batch {batch_index + 1}/{len(dataloader)} "
                f"({sample_count} images)."
            )

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
    for stage_index, second_moment in enumerate(teacher_second_moments):
        centered = second_moment - torch.outer(
            teacher_sums[stage_index], teacher_sums[stage_index]
        ) / float(sample_count)
        teacher_centered_self.append(centered)

    for stage_index, second_moment in enumerate(student_second_moments):
        centered = second_moment - torch.outer(
            student_sums[stage_index], student_sums[stage_index]
        ) / float(sample_count)
        student_centered_self.append(centered)

    # Convert stage widths to slice boundaries so each heatmap cell receives the
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
                f"Teacher stage {teacher_index + 1} has zero centered variance; "
                "linear CKA is undefined for that stage."
            )

        teacher_start = teacher_offsets[teacher_index]
        teacher_end = teacher_offsets[teacher_index + 1]

        for student_index in range(len(student_widths)):
            student_self_energy = torch.sum(
                student_centered_self[student_index] ** 2
            )
            if student_self_energy <= 0:
                raise ValueError(
                    f"Student stage {student_index + 1} has zero centered variance; "
                    "linear CKA is undefined for that stage."
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
    while the PNG provides the stage-by-stage visual summary typically used to
    inspect teacher-student representation alignment.

    Steps
    -----
    1. Create ``output_dir`` if it does not already exist.
    2. Write the matrix to ``cka_matrix.csv`` with stage labels on both axes.
    3. Render a fixed-range ``[0, 1]`` heatmap and annotate every cell.
    4. Save the figure to ``cka_heatmap.png``.

    Parameters
    ----------
    cka_matrix : CKAMatrix
        Two-dimensional CKA array with teacher stages on rows and student stages
        on columns.
    teacher_labels : List[str]
        Row labels matching the teacher-stage dimension.
    student_labels : List[str]
        Column labels matching the student-stage dimension.
    output_dir : str
        Directory where CSV and PNG outputs are written.

    Returns
    -------
    Tuple[str, str]
        Paths to the saved CSV file and heatmap PNG, respectively.

    Raises
    ------
    ValueError
        If the matrix shape does not match the supplied stage labels.
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
    cka_df.to_csv(csv_path, index_label="teacher_stage")

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
    ax.set_xlabel("Student encoder stages")
    ax.set_ylabel("Teacher encoder stages")
    ax.set_title("Teacher-Student Linear CKA")

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

    The teacher and student checkpoints are mandatory. Optional arguments let
    the caller control DataLoader memory/throughput, cap the deterministic test
    subset for quick experiments, and choose the result directory.

    Steps
    -----
    1. Require separate teacher and student checkpoint paths.
    2. Add optional batch-size, maximum-sample, and output-directory controls.
    3. Parse and return the command-line namespace.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments containing ``teacher``, ``student``,
        ``batch_size``, ``max_samples``, and ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare teacher and student encoder representations with "
            "stage-wise linear Centered Kernel Alignment (CKA)."
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
    used for predictive evaluation. Only encoder activations are needed, and
    global average pooling plus streaming sufficient statistics keeps the
    analysis practical for ResNet-18 and MobileNetV2.

    Steps
    -----
    1. Parse teacher/student checkpoint paths and runtime options.
    2. Load both serialized models on the configured evaluation device.
    3. Recreate the deterministic BDD100k held-out test split and preprocessing.
    4. Optionally take the first ``max_samples`` rows for a reproducible subset.
    5. Compute all encoder-stage pair CKA scores in one dataset pass.
    6. Save the numeric matrix and annotated heatmap and print their locations.

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

    print("Computing stage-wise linear CKA...")
    cka_matrix, teacher_labels, student_labels, sample_count = compute_linear_cka_matrix(
        teacher=teacher,
        student=student,
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
