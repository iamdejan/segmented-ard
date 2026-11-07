"""Robustness evaluation script for semantic segmentation models under PGD attack.

This module evaluates the adversarial robustness of trained PyTorch segmentation
models (.pt files) using a multi-step, multi-restart Projected Gradient Descent (PGD)
untargeted attack. It computes clean vs. adversarial segmentation metrics while
excluding the Unknown class and classes that have no ground-truth support from macro
averages.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple, Union, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, UInt8, jaxtyped
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Shape aliases enforced at runtime by jaxtyped + beartype to eliminate
# broadcast shape mismatches across the evaluation pipeline.
ImageTensor = Float[Tensor, "3 h w"]  # Single image, channel-first layout
MaskTensor = Float[Tensor, "c h w"]  # One-hot mask, c == NUM_CLASSES
BatchImage = Float[Tensor, "b 3 h w"]  # Collated batch of images
BatchMask = Float[Tensor, "b c h w"]  # Collated batch of one-hot masks
Logits = Float[Tensor, "b c h w"]  # Model output, c == NUM_CLASSES
NumpyImage = Float[np.ndarray, "h w 3"]  # Single image in channels-last layout
ClassIndexArray = UInt8[np.ndarray, "h w"]  # Per-pixel class index map


# BDD100k color-label palette for rendering masks. Row indices match class IDs (0-19).
CLASS_COLORS = np.array([
    [128,  64, 128],   # 0  - Road
    [244,  35, 232],   # 1  - Sidewalk
    [ 70,  70,  70],   # 2  - Building
    [102, 102, 156],   # 3  - Wall
    [190, 153, 153],   # 4  - Fence
    [153, 153, 153],   # 5  - Pole
    [250, 170,  30],   # 6  - Traffic Light
    [220, 220,   0],   # 7  - Traffic Sign
    [107, 142,  35],   # 8  - Vegetation
    [152, 251, 152],   # 9  - Terrain
    [ 70, 130, 180],   # 10 - Sky
    [220,  20,  60],   # 11 - Person
    [255,   0,   0],   # 12 - Rider
    [  0,   0, 142],   # 13 - Car
    [  0,   0,  70],   # 14 - Truck
    [  0,  60, 100],   # 15 - Bus
    [  0,  80, 100],   # 16 - Train
    [  0,   0, 230],   # 17 - Motorcycle
    [119,  11,  32],   # 18 - Bicycle
    [  0,   0,   0],   # 19 - Unknown
], dtype=np.uint8)

# Human-readable labels mapped to class indices.
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "unknown",
]


class Configuration:
    """Evaluation constants and dataset image dimension specifications."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 20
    NUM_VALID_CLASSES = 19  # Classes 0 to 18 (class 19 is Unknown)
    IGNORED_CLASS = 19
    METRIC_IGNORE_INDEX = -1

    BATCH_SIZE = 16 if torch.cuda.device_count() < 2 else (16 * torch.cuda.device_count())
    NUM_WORKERS = 2
    SEED = 768

    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 640
    CHANNELS = 3


class Path:
    """Filesystem directory locations for BDD100k images and masks."""

    BASE = "./data/bdd100k"
    SEGMENTATION_MASK_LABEL_FOLDER = BASE + "/segmentation_maps/color_labels"
    SEGMENTATION_MASK_TRAIN_PATH = SEGMENTATION_MASK_LABEL_FOLDER + "/train"
    SEGMENTATION_MASK_VAL_PATH = SEGMENTATION_MASK_LABEL_FOLDER + "/val"

    IMAGE_FOLDER = BASE + "/images_10k"
    IMAGE_TRAIN_PATH = IMAGE_FOLDER + "/train"
    IMAGE_VAL_PATH = IMAGE_FOLDER + "/val"


def color_label_to_class_index(label: np.ndarray) -> np.ndarray:
    """Map an RGB color-label image to a per-pixel class-index map.

    BDD100k stores masks as RGB PNGs matching ``CLASS_COLORS``. Semantic
    segmentation loss functions and metrics require 2D class index maps
    rather than RGB triples, so each pixel is mapped to its palette row index.

    Steps
    -----
    1. Initialize output array with the fallback ID for the ``Unknown`` class.
    2. Vectorially match each RGB triple in ``CLASS_COLORS`` and assign the class ID.

    Parameters
    ----------
    label : np.ndarray
        RGB color-label array of shape ``(H, W, 3)``.

    Returns
    -------
    np.ndarray
        Class-index array of shape ``(H, W)`` with values in ``[0, NUM_CLASSES)``.
    """
    class_ids = np.full(label.shape[:2], Configuration.IGNORED_CLASS, dtype=np.uint8)
    for class_id, (red, green, blue) in enumerate(CLASS_COLORS):
        match = (
            (label[..., 0] == red)
            & (label[..., 1] == green)
            & (label[..., 2] == blue)
        )
        class_ids[match] = class_id
    return class_ids


def find_image_path_from_mask(complete_mask_path: str, base_image_path: str) -> str:
    """Derive the corresponding JPEG image path from a mask PNG path.

    Steps
    -----
    1. Extract the image ID by stripping the mask-specific suffix.
    2. Construct and return the full path in the target image folder.

    Parameters
    ----------
    complete_mask_path : str
        Full filesystem path to a segmentation mask PNG.
    base_image_path : str
        Directory containing the corresponding JPEG images.

    Returns
    -------
    str
        Full filesystem path to the matching image file.
    """
    file_path_split = complete_mask_path.split("/")
    mask_file_name = file_path_split[-1].split("_")[0]

    image_path = base_image_path + "/" + mask_file_name + ".jpg"
    return image_path


def find_train_image_path_from_mask(complete_mask_path: str) -> str:
    """Locate the training image corresponding to a training mask.

    Steps
    -----
    1. Delegate to :func:`find_image_path_from_mask` using
       ``Path.IMAGE_TRAIN_PATH``.

    Parameters
    ----------
    complete_mask_path : str
        Full path to a training mask PNG.

    Returns
    -------
    str
        Full path to the corresponding training image JPEG.
    """
    return find_image_path_from_mask(complete_mask_path, Path.IMAGE_TRAIN_PATH)


def find_val_image_path_from_mask(complete_mask_path: str) -> str:
    """Locate the validation image corresponding to a validation mask.

    Steps
    -----
    1. Delegate to :func:`find_image_path_from_mask` using
       ``Path.IMAGE_VAL_PATH``.

    Parameters
    ----------
    complete_mask_path : str
        Full path to a validation mask PNG.

    Returns
    -------
    str
        Full path to the corresponding validation image JPEG.
    """
    return find_image_path_from_mask(complete_mask_path, Path.IMAGE_VAL_PATH)


def load_dataset_from_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan disk and load the BDD100k train, validation, and test splits.

    This function duplicates the exact partitioning logic from the training
    scripts so the model is evaluated on the identical test split that was held
    out during training.

    Steps
    -----
    1. Collect train masks and filter non-standard resolutions without modifying
       candidate lists during iteration.
    2. Map to corresponding train images and filter non-matching pairs.
    3. Partition the train-test pool using 80/20 ``train_test_split`` with the
       fixed ``Configuration.SEED``.
    4. Collect and validate validation masks and images without in-place mutation.
    5. Return DataFrames for train, validation, and test splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames representing ``(train_df, val_df, test_df)``.
    """
    # Build the valid training pool without mutating a list during iteration.
    # In-place deletion during iteration skips adjacent candidate elements.
    train_mask_candidates = glob.glob(f"{Path.SEGMENTATION_MASK_TRAIN_PATH}/*.png")
    train_mask_paths: List[str] = []
    problematic_masks: List[str] = []
    for complete_mask_path in train_mask_candidates:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_masks.append(complete_mask_path)
                continue
        train_mask_paths.append(complete_mask_path)
    print(f"Problematic masks: {problematic_masks}")

    train_image_paths: List[str] = []
    validated_train_mask_paths: List[str] = []
    problematic_images: List[str] = []
    for complete_mask_path in train_mask_paths:
        complete_image_path = find_train_image_path_from_mask(complete_mask_path)
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_images.append(complete_image_path)
                continue
        train_image_paths.append(complete_image_path)
        validated_train_mask_paths.append(complete_mask_path)
    print(f"Problematic images: {problematic_images}")

    train_test_df = pd.DataFrame({
        "image_paths": train_image_paths,
        "mask_paths": validated_train_mask_paths,
    })
    train_df, test_df = train_test_split(
        train_test_df, test_size=0.2, random_state=Configuration.SEED
    )

    # Load the BDD100k validation split, again filtering non-standard samples
    # without mutating candidate lists during iteration.
    val_mask_candidates = glob.glob(f"{Path.SEGMENTATION_MASK_VAL_PATH}/*.png")
    val_mask_paths: List[str] = []
    problematic_val_masks: List[str] = []
    for complete_mask_path in val_mask_candidates:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_val_masks.append(complete_mask_path)
                continue
        val_mask_paths.append(complete_mask_path)
    print(f"Problematic val masks: {problematic_val_masks}")

    val_image_paths: List[str] = []
    validated_val_mask_paths: List[str] = []
    problematic_val_images: List[str] = []
    for complete_mask_path in val_mask_paths:
        complete_image_path = find_val_image_path_from_mask(complete_mask_path)
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_val_images.append(complete_image_path)
                continue
        val_image_paths.append(complete_image_path)
        validated_val_mask_paths.append(complete_mask_path)
    print(f"Problematic val images: {problematic_val_images}")

    val_df = pd.DataFrame({
        "image_paths": val_image_paths,
        "mask_paths": validated_val_mask_paths,
    })

    return train_df, val_df, test_df

class BDDSegmentationDataset(Dataset[Tuple[ImageTensor, MaskTensor]]):
    """BDD100k semantic segmentation dataset loader.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``image_paths`` and ``mask_paths`` columns.
    transform : Optional[A.Compose], optional
        Albumentations transform pipeline applied to image and mask. Defaults to None.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[A.Compose] = None) -> None:
        super().__init__()
        self.image_paths: List[str] = df["image_paths"].to_list()
        self.mask_paths: List[str] = df["mask_paths"].to_list()
        self.transform = transform

    @jaxtyped(typechecker=beartype)
    def load_sample(self, index: int) -> Tuple[NumpyImage, ClassIndexArray]:
        """Load an image and its class-index mask from disk.

        Steps
        -----
        1. Open image and mask PNGs as RGB arrays.
        2. Normalize image pixel intensities to [0, 1].
        3. Convert RGB color-label mask to per-pixel class index array.

        Parameters
        ----------
        index : int
            Index of the sample to load.

        Returns
        -------
        Tuple[NumpyImage, ClassIndexArray]
            Normalized image array and per-pixel uint8 class index map.

        Raises
        ------
        IndexError
            If ``index`` is out of bounds for the dataset path lists.
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("RGB")

        image = np.array(image_pil).astype(np.float32) / 255.0
        class_mask = color_label_to_class_index(np.array(mask_pil))

        return image, class_mask

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> Tuple[ImageTensor, MaskTensor]:
        """Retrieve transformed image and one-hot mask tensors at index.

        Steps
        -----
        1. Load raw sample via ``load_sample``.
        2. Apply resize and PyTorch transposition transforms.
        3. One-hot encode mask into ``(NUM_CLASSES, H, W)`` float tensor.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[ImageTensor, MaskTensor]
            Pair of transformed image ``(3, H, W)`` and one-hot mask ``(20, H, W)``.
        """
        image, class_mask = self.load_sample(index)

        if self.transform:
            transformed = self.transform(image=image, mask=class_mask)
        else:
            transformed = ToTensorV2()(image=image, mask=class_mask)

        # One-hot encode so the channel dimension aligns with the model's logits
        mask_one_hot = torch.nn.functional.one_hot(
            transformed["mask"].to(torch.int64), Configuration.NUM_CLASSES
        ).permute(2, 0, 1).float()

        return transformed["image"], mask_one_hot


@jaxtyped(typechecker=beartype)
def forward(model: nn.Module, x: BatchImage) -> Logits:
    """Execute model forward pass and verify batch tensor shape contract.

    Parameters
    ----------
    model : nn.Module
        Trained semantic segmentation model.
    x : BatchImage
        Batch of images of shape ``(B, 3, H, W)``.

    Returns
    -------
    Logits
        Raw output logits of shape ``(B, NUM_CLASSES, H, W)``.
    """
    return cast(Logits, model(x))


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def compute_batch_macro_iou(
    y_pred: Logits,
    y_true: BatchMask,
    num_classes: int = 19,
    ignored_class: int = 19,
    eps: float = 1e-7,
) -> float:
    """Compute per-batch macro IoU over classes present in ground truth.

    Steps
    -----
    1. Collapse predictions and targets to 2D class maps using argmax.
    2. Exclude pixels whose ground-truth class is ``ignored_class``.
    3. Evaluate IoU for classes 0 to ``num_classes - 1`` that occur in ground truth.
    4. Average IoU over that fixed ground-truth-supported class set so predictions
       cannot change which classes contribute to the metric.

    Parameters
    ----------
    y_pred : Logits
        Model output logits of shape ``(B, C, H, W)``.
    y_true : BatchMask
        One-hot ground-truth masks of shape ``(B, C, H, W)``.
    num_classes : int, optional
        Number of valid classes to consider. Defaults to 19.
    ignored_class : int, optional
        Ground-truth class excluded from evaluation. Defaults to 19 (Unknown).
    eps : float, optional
        Smoothing constant to prevent division by zero. Defaults to 1e-7.

    Returns
    -------
    float
        Macro IoU score averaged across ground-truth-supported classes in the batch.
    """
    preds = y_pred.argmax(dim=1)
    targets = y_true.argmax(dim=1)
    valid_pixels = targets != ignored_class

    iou_per_class: List[float] = []
    for cls in range(num_classes):
        # Apply the target-derived validity mask to both tensors. Unknown target
        # pixels therefore cannot create false positives for otherwise valid classes.
        pred_mask = (preds == cls) & valid_pixels
        true_mask = (targets == cls) & valid_pixels

        # The macro class set must depend only on ground truth. If prediction
        # support were allowed to decide inclusion, clean and adversarial scores
        # could average over different class sets.
        if not true_mask.any().item():
            continue

        intersection = (pred_mask & true_mask).sum().float().item()
        union = (pred_mask | true_mask).sum().float().item()

        iou_per_class.append((intersection + eps) / (union + eps))

    return float(np.mean(iou_per_class)) if iou_per_class else 0.0


def colorize_mask(class_mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map a 2D class-index array to an RGB image using the specified color palette.

    Steps
    -----
    1. Cast the class index array to integer indices.
    2. Index rows of ``palette`` to obtain an RGB representation.

    Parameters
    ----------
    class_mask : np.ndarray
        Array of shape ``(H, W)`` containing integer class indices.
    palette : np.ndarray
        Array of shape ``(NUM_CLASSES, 3)`` mapping class ID to RGB color.

    Returns
    -------
    np.ndarray
        RGB image of shape ``(H, W, 3)`` and dtype matching ``palette``.
    """
    return palette[class_mask.astype(np.int64)]


def load_trained_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a trained segmentation model checkpoint from disk.

    Steps
    -----
    1. Resolve and verify checkpoint path existence.
    2. Load model using ``torch.load`` with ``weights_only=False`` to unpack
       serialized ``nn.Module`` objects produced by ``train_teacher.py``/``train_student.py``.
    3. Transfer model to target ``device`` and switch into ``eval()`` mode.
    4. Freeze all model parameters to prevent weight updates during adversarial attacks.

    Parameters
    ----------
    model_path : str
        Filesystem path to the model ``.pt`` file.
    device : torch.device
        Device (CUDA or CPU) on which to place the model.

    Returns
    -------
    nn.Module
        Loaded, frozen PyTorch segmentation model ready for evaluation.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    TypeError
        If the loaded checkpoint object is not an ``nn.Module``.
    """
    # weights_only=False is required because the project saves entire nn.Module objects
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, nn.Module):
        raise TypeError(
            f"Checkpoint at '{model_path}' loaded an object of type {type(checkpoint)}, "
            "but an nn.Module instance was expected."
        )

    model = checkpoint.to(device)
    model.eval()

    # Freeze model weights so autograd does not allocate parameter gradient memory
    for param in model.parameters():
        param.requires_grad = False

    return model


@jaxtyped(typechecker=beartype)
def pgd_attack(
    model: nn.Module,
    images: BatchImage,
    targets: BatchMask,
    loss_fn: Optional[nn.Module] = None,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_steps: int = 10,
    random_start: bool = True,
    num_restarts: int = 5,
) -> BatchImage:
    """Generate worst-of-restarts untargeted PGD adversarial perturbations.

    PGD iteratively maximizes the segmentation loss with respect to the input image
    within an L-infinity ball of radius ``epsilon``. Multiple random restarts reduce
    sensitivity to a single attack trajectory. The clean image is included as a
    baseline candidate, and the returned batch keeps the candidate with the highest
    valid-pixel cross-entropy independently for each sample.

    Steps
    -----
    1. Freeze model in eval mode and prepare target class index map.
    2. Score the clean image using per-sample cross-entropy on non-Unknown pixels.
    3. For every restart, initialize a candidate inside the epsilon ball and run
       ``num_steps`` projected gradient-ascent updates.
    4. Score each final candidate per sample and retain it only when its
       cross-entropy exceeds the current worst case.
    5. Return the per-sample worst candidate across the clean baseline and restarts.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model being attacked.
    images : BatchImage
        Batch of clean images with shape ``(B, 3, H, W)`` and values in ``[0, 1]``.
    targets : BatchMask
        One-hot ground-truth masks of shape ``(B, NUM_CLASSES, H, W)``.
    loss_fn : Optional[nn.Module], optional
        Differentiable loss function to maximize. Defaults to
        ``nn.CrossEntropyLoss(ignore_index=19)`` so unannotated pixels do not
        corrupt adversarial gradients.
    epsilon : float, optional
        Maximum L-infinity perturbation radius. Defaults to 0.03.
    alpha : float, optional
        Step size per iteration. Defaults to 0.01.
    num_steps : int, optional
        Number of attack iterations. Defaults to 10.
    random_start : bool, optional
        Whether each restart begins from a random point in the epsilon ball.
        Defaults to True.
    num_restarts : int, optional
        Number of independent PGD trajectories. Defaults to 5.

    Returns
    -------
    BatchImage
        Adversarial image batch of shape ``(B, 3, H, W)``.

    Raises
    ------
    ValueError
        If attack hyperparameters are outside their valid ranges.
    """
    model.eval()

    # Reject invalid attack configurations here as well as in the CLI so this
    # function remains safe when imported and called directly by another module.
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    if alpha <= 0.0:
        raise ValueError("alpha must be greater than zero.")
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero.")
    if num_restarts <= 0:
        raise ValueError("num_restarts must be greater than zero.")

    # Ignore class 19 (Unknown) so background noise does not misguide the attack
    if loss_fn is None:
        effective_loss: nn.Module = nn.CrossEntropyLoss(ignore_index=Configuration.IGNORED_CLASS)
    else:
        effective_loss = loss_fn

    # Collapse one-hot targets to class indices for standard cross-entropy evaluation
    target_indices = targets.argmax(dim=1).to(torch.int64)

    # Use unreduced cross-entropy only for restart selection. This gives every
    # sample its own attack score instead of selecting one restart for the batch.
    selection_loss = nn.CrossEntropyLoss(
        ignore_index=Configuration.IGNORED_CLASS,
        reduction="none",
    )
    valid_pixels = target_indices != Configuration.IGNORED_CLASS
    if not valid_pixels.any().item():
        # There is no supervised signal to maximize when the whole batch is
        # Unknown, so the clean image is the only meaningful fallback candidate.
        return cast(BatchImage, images.clone().detach())
    valid_counts = valid_pixels.sum(dim=(1, 2)).clamp_min(1)

    # The clean input belongs to the admissible epsilon ball, so it is a valid
    # baseline for worst-case selection. Keeping it prevents the evaluator from
    # returning a restart whose attack objective is weaker than the clean input.
    with torch.no_grad():
        clean_logits = forward(model, images)
        clean_pixel_loss = selection_loss(clean_logits, target_indices)
        best_loss = (clean_pixel_loss * valid_pixels).sum(dim=(1, 2)) / valid_counts
    best_images = images.clone().detach()

    for _ in range(num_restarts):
        # Independent random starts make PGD less sensitive to local geometry of
        # robust models. A caller can disable them for deterministic ablations.
        if random_start:
            noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(images + noise, 0.0, 1.0).detach()
        else:
            x_adv = images.clone().detach()

        # Multi-step projected gradient ascent maximizes the requested attack
        # loss while preserving both the L-infinity and valid-pixel constraints.
        for _ in range(num_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            adv_logits = forward(model, x_adv)
            loss = effective_loss(adv_logits, target_indices)
            grad = torch.autograd.grad(
                loss,
                x_adv,
                retain_graph=False,
                create_graph=False,
            )[0]

            perturbation = x_adv.detach() + alpha * grad.sign() - images
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            x_adv = torch.clamp(images + perturbation, 0.0, 1.0).detach()

        # Score the final restart per sample. The boolean mask is broadcast over
        # image channels and spatial dimensions so each batch item is replaced
        # independently rather than choosing a single restart for the whole batch.
        with torch.no_grad():
            restart_logits = forward(model, x_adv)
            restart_pixel_loss = selection_loss(restart_logits, target_indices)
            restart_loss = (
                (restart_pixel_loss * valid_pixels).sum(dim=(1, 2)) / valid_counts
            )
        replace = restart_loss > best_loss
        best_images = torch.where(
            replace[:, None, None, None],
            x_adv,
            best_images,
        )
        best_loss = torch.maximum(best_loss, restart_loss)

    return cast(BatchImage, best_images)


def evaluate_robustness(
    model: nn.Module,
    dataloader: DataLoader[Tuple[ImageTensor, MaskTensor]],
    device: torch.device,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_steps: int = 10,
    num_classes: int = 19,
    ignore_index: int = -1,
    ignored_class: int = 19,
    num_restarts: int = 5,
) -> Dict[str, Union[float, List[float]]]:
    """Run clean and multi-restart PGD adversarial evaluation on a dataloader.

    Confusion counts (TP, FP, FN, TN) are accumulated over the entire test set
    before reduction, guaranteeing statistically robust metrics regardless of
    batch-level class imbalances.

    Steps
    -----
    1. Iterate through dataloader batches.
    2. Run clean forward pass under ``torch.no_grad`` to obtain clean logits.
    3. Generate worst-of-restarts PGD adversarial images using ``pgd_attack``.
    4. Run adversarial forward pass under ``torch.no_grad`` to obtain perturbed logits.
    5. Ignore Unknown ground-truth pixels while keeping Unknown predictions as
       errors on valid ground-truth pixels.
    6. Accumulate per-class confusion statistics for both clean and adversarial outputs.
    7. Compute macro IoU and Dice over only classes with ground-truth support,
       plus globally pooled pixel accuracy, per-class IoU, and valid-pixel
       cross-entropy for attack-strength diagnostics.

    Parameters
    ----------
    model : nn.Module
        Trained model being evaluated.
    dataloader : DataLoader[Tuple[ImageTensor, MaskTensor]]
        DataLoader delivering evaluation batches.
    device : torch.device
        Hardware device for computation.
    epsilon : float, optional
        L-infinity perturbation bound. Defaults to 0.03.
    alpha : float, optional
        Step size per iteration. Defaults to 0.01.
    num_steps : int, optional
        Number of PGD iterations. Defaults to 10.
    num_classes : int, optional
        Number of valid segmentation classes (excluding Unknown). Defaults to 19.
    ignore_index : int, optional
        Sentinel index for ignored pixels in ``smp.metrics.get_stats``. Defaults to -1.
    ignored_class : int, optional
        Class index corresponding to Unknown in dataset. Defaults to 19.
    num_restarts : int, optional
        Number of independent random PGD restarts. Defaults to 5.

    Returns
    -------
    Dict[str, Union[float, List[float]]]
        Dictionary with clean/adversarial IoU, pixel accuracy, macro Dice,
        cross-entropy diagnostics, batch macro IoU, and per-class IoU values.

    Raises
    ------
    ValueError
        If the dataloader produces zero batches or contains no valid ground-truth class.
    """
    model.eval()

    clean_tp_total: Optional[torch.Tensor] = None
    clean_fp_total: Optional[torch.Tensor] = None
    clean_fn_total: Optional[torch.Tensor] = None
    clean_tn_total: Optional[torch.Tensor] = None

    adv_tp_total: Optional[torch.Tensor] = None
    adv_fp_total: Optional[torch.Tensor] = None
    adv_fn_total: Optional[torch.Tensor] = None
    adv_tn_total: Optional[torch.Tensor] = None

    clean_batch_iou_sum = 0.0
    adv_batch_iou_sum = 0.0
    clean_cross_entropy_sum = 0.0
    adv_cross_entropy_sum = 0.0
    valid_pixel_count = 0

    # Report the same cross-entropy objective used to select the worst PGD
    # restart. Summing here permits an exact valid-pixel dataset average later.
    cross_entropy_sum = nn.CrossEntropyLoss(
        ignore_index=ignored_class,
        reduction="sum",
    )

    for X, y in tqdm(dataloader, desc="Evaluating robustness"):
        X, y = X.to(device), y.to(device)

        # 1. Clean forward pass (no gradients needed)
        with torch.no_grad():
            clean_logits = forward(model, X)
            clean_preds = clean_logits.argmax(dim=1)

        # 2. Craft worst-of-restarts PGD adversarial examples.
        adv_X = pgd_attack(
            model=model,
            images=X,
            targets=y,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            random_start=True,
            num_restarts=num_restarts,
        )

        # 3. Adversarial forward pass
        with torch.no_grad():
            adv_logits = forward(model, adv_X)
            adv_preds = adv_logits.argmax(dim=1)

        # 4. Ignore only Unknown ground-truth pixels. Unknown predictions on
        # valid labels must remain predictions so they correctly contribute a
        # false negative to the corresponding ground-truth class.
        targets = y.argmax(dim=1)
        valid_pixel_count += int((targets != ignored_class).sum().item())
        with torch.no_grad():
            clean_cross_entropy_sum += float(
                cross_entropy_sum(clean_logits, targets).item()
            )
            adv_cross_entropy_sum += float(
                cross_entropy_sum(adv_logits, targets).item()
            )
        target_relabelled = torch.where(
            targets == ignored_class, torch.full_like(targets, ignore_index), targets
        )

        # SMP requires predictions to stay inside [0, num_classes_for_stats).
        # Include the ignored prediction class while constructing statistics,
        # then discard its own class row after valid-class false negatives have
        # already been counted.
        num_classes_for_stats = max(num_classes, ignored_class + 1)

        # 5. Accumulate clean confusion counts
        c_tp, c_fp, c_fn, c_tn = smp.metrics.functional.get_stats(
            clean_preds,
            target_relabelled,
            mode="multiclass",
            ignore_index=ignore_index,
            num_classes=num_classes_for_stats,
        )
        c_tp = c_tp[:, :num_classes]
        c_fp = c_fp[:, :num_classes]
        c_fn = c_fn[:, :num_classes]
        c_tn = c_tn[:, :num_classes]
        clean_tp_total = c_tp.sum(dim=0) if clean_tp_total is None else clean_tp_total + c_tp.sum(dim=0)
        clean_fp_total = c_fp.sum(dim=0) if clean_fp_total is None else clean_fp_total + c_fp.sum(dim=0)
        clean_fn_total = c_fn.sum(dim=0) if clean_fn_total is None else clean_fn_total + c_fn.sum(dim=0)
        clean_tn_total = c_tn.sum(dim=0) if clean_tn_total is None else clean_tn_total + c_tn.sum(dim=0)

        # 6. Accumulate adversarial confusion counts
        a_tp, a_fp, a_fn, a_tn = smp.metrics.functional.get_stats(
            adv_preds,
            target_relabelled,
            mode="multiclass",
            ignore_index=ignore_index,
            num_classes=num_classes_for_stats,
        )
        a_tp = a_tp[:, :num_classes]
        a_fp = a_fp[:, :num_classes]
        a_fn = a_fn[:, :num_classes]
        a_tn = a_tn[:, :num_classes]
        adv_tp_total = a_tp.sum(dim=0) if adv_tp_total is None else adv_tp_total + a_tp.sum(dim=0)
        adv_fp_total = a_fp.sum(dim=0) if adv_fp_total is None else adv_fp_total + a_fp.sum(dim=0)
        adv_fn_total = a_fn.sum(dim=0) if adv_fn_total is None else adv_fn_total + a_fn.sum(dim=0)
        adv_tn_total = a_tn.sum(dim=0) if adv_tn_total is None else adv_tn_total + a_tn.sum(dim=0)

        # Per-batch macro IoU tracking
        clean_batch_iou_sum += compute_batch_macro_iou(
            clean_logits,
            y,
            num_classes=num_classes,
            ignored_class=ignored_class,
        )
        adv_batch_iou_sum += compute_batch_macro_iou(
            adv_logits,
            y,
            num_classes=num_classes,
            ignored_class=ignored_class,
        )

    if clean_tp_total is None or adv_tp_total is None:
        raise ValueError("The evaluation dataloader yielded zero batches.")
    if valid_pixel_count == 0:
        raise ValueError("The evaluation data contains no valid ground-truth pixels.")

    # Ground-truth support is prediction-independent, so clean and adversarial
    # macro metrics use exactly the same class set. This prevents an absent class
    # from receiving SMP's default perfect score and inflating adversarial Dice.
    present_in_ground_truth = (clean_tp_total + clean_fn_total) > 0
    if not present_in_ground_truth.any().item():
        raise ValueError("The evaluation data contains no valid ground-truth classes.")

    # Preserve the class dimension by requesting unreduced scores. Explicitly
    # map 0/0 cases to zero; absent classes are subsequently excluded from macro
    # averaging but remain unambiguous in per-class CSV output.
    clean_class_iou_tensor = smp.metrics.iou_score(
        clean_tp_total,
        clean_fp_total,
        clean_fn_total,
        clean_tn_total,
        reduction=None,
        zero_division=0.0,
    )
    adv_class_iou_tensor = smp.metrics.iou_score(
        adv_tp_total,
        adv_fp_total,
        adv_fn_total,
        adv_tn_total,
        reduction=None,
        zero_division=0.0,
    )
    clean_class_iou = cast(List[float], clean_class_iou_tensor.tolist())
    adv_class_iou = cast(List[float], adv_class_iou_tensor.tolist())

    # Dataset-wide macro IoU is the arithmetic mean of per-class IoU only for
    # classes that actually occur in the ground truth of this evaluation set.
    clean_mean_iou = float(clean_class_iou_tensor[present_in_ground_truth].mean().item())
    adv_mean_iou = float(adv_class_iou_tensor[present_in_ground_truth].mean().item())

    # Pixel accuracy is pooled globally across all valid classes.
    clean_pixel_acc = float(
        smp.metrics.recall(
            clean_tp_total,
            clean_fp_total,
            clean_fn_total,
            clean_tn_total,
            reduction="micro",
            zero_division=0.0,
        ).item()
    )
    adv_pixel_acc = float(
        smp.metrics.recall(
            adv_tp_total,
            adv_fp_total,
            adv_fn_total,
            adv_tn_total,
            reduction="micro",
            zero_division=0.0,
        ).item()
    )

    # Calculate class-wise Dice first so the class axis is preserved. Calling
    # SMP with reduction="macro" on these 1-D per-class totals would sum along
    # dim=0 before scoring, which does not produce the intended macro average.
    clean_class_dice_tensor = smp.metrics.f1_score(
        clean_tp_total,
        clean_fp_total,
        clean_fn_total,
        clean_tn_total,
        reduction=None,
        zero_division=0.0,
    )
    adv_class_dice_tensor = smp.metrics.f1_score(
        adv_tp_total,
        adv_fp_total,
        adv_fn_total,
        adv_tn_total,
        reduction=None,
        zero_division=0.0,
    )

    # Average the same ground-truth-supported class set for clean and adversarial
    # Dice. This removes the absent-class 0/0 -> 1 artifact from macro scores.
    clean_dice = float(clean_class_dice_tensor[present_in_ground_truth].mean().item())
    adv_dice = float(adv_class_dice_tensor[present_in_ground_truth].mean().item())
    clean_cross_entropy = clean_cross_entropy_sum / valid_pixel_count
    adv_cross_entropy = adv_cross_entropy_sum / valid_pixel_count

    num_batches = len(dataloader)
    return {
        "clean_mean_iou": clean_mean_iou,
        "adv_mean_iou": adv_mean_iou,
        "clean_pixel_accuracy": clean_pixel_acc,
        "adv_pixel_accuracy": adv_pixel_acc,
        "clean_dice": clean_dice,
        "adv_dice": adv_dice,
        "clean_cross_entropy": clean_cross_entropy,
        "adv_cross_entropy": adv_cross_entropy,
        "clean_class_iou": clean_class_iou,
        "adv_class_iou": adv_class_iou,
        "clean_batch_macro_iou": clean_batch_iou_sum / num_batches,
        "adv_batch_macro_iou": adv_batch_iou_sum / num_batches,
    }


def visualize_adversarial_predictions(
    model: nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_steps: int = 10,
    num_samples: int = 4,
    output_path: str = "./model/adversarial_predictions.png",
    num_restarts: int = 5,
) -> None:
    """Render a visual side-by-side comparison of clean and adversarial predictions.

    Steps
    -----
    1. Sample random test rows using the fixed seed.
    2. Craft worst-of-restarts PGD adversarial examples for each sample.
    3. Generate colorized prediction maps for clean and adversarial inputs.
    4. Export comparison grid (Image / Ground Truth / Clean Pred / Adv Pred / Perturbation).

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    test_df : pd.DataFrame
        DataFrame of test samples.
    device : torch.device
        Hardware device for inference.
    epsilon : float, optional
        L-infinity perturbation bound. Defaults to 0.03.
    alpha : float, optional
        Step size per iteration. Defaults to 0.01.
    num_steps : int, optional
        Number of PGD iterations. Defaults to 10.
    num_samples : int, optional
        Number of samples to visualize. Defaults to 4.
    output_path : str, optional
        Destination image path.
    num_restarts : int, optional
        Number of independent random PGD restarts. Defaults to 5.

    Raises
    ------
    ValueError
        If ``test_df`` is empty.
    """
    if test_df.empty:
        raise ValueError("test_df contains no rows to visualize.")

    sample_df = test_df.sample(
        n=min(num_samples, len(test_df)),
        random_state=Configuration.SEED,
    ).reset_index(drop=True)

    num_rows = len(sample_df)
    fig, axes = plt.subplots(num_rows, 5, figsize=(25, 5 * num_rows))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    sample_ds = BDDSegmentationDataset(sample_df)
    model.eval()

    for row in range(num_rows):
        image, class_mask = sample_ds.load_sample(row)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).contiguous().unsqueeze(0).to(device)
        mask_tensor = torch.nn.functional.one_hot(
            torch.from_numpy(class_mask).to(torch.int64), Configuration.NUM_CLASSES
        ).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # Clean inference
        with torch.no_grad():
            clean_logits = model(image_tensor)
        clean_pred_class = clean_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Adversarial attack and inference
        adv_image_tensor = pgd_attack(
            model=model,
            images=image_tensor,
            targets=mask_tensor,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            random_start=True,
            num_restarts=num_restarts,
        )
        with torch.no_grad():
            adv_logits = model(adv_image_tensor)
        adv_pred_class = adv_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Colorize masks
        clean_pred_color = colorize_mask(clean_pred_class, CLASS_COLORS).astype(np.float32) / 255.0
        adv_pred_color = colorize_mask(adv_pred_class, CLASS_COLORS).astype(np.float32) / 255.0
        true_mask_color = colorize_mask(class_mask, CLASS_COLORS).astype(np.float32) / 255.0

        # Amplified perturbation map for visual inspection
        diff = np.abs(adv_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() - image)
        diff_vis = np.clip(diff * 10.0, 0.0, 1.0)

        # Plot columns
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Original Clean Image")

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(true_mask_color, alpha=0.5)
        axes[row, 1].set_title("Ground Truth Mask")

        axes[row, 2].imshow(image)
        axes[row, 2].imshow(clean_pred_color, alpha=0.5)
        axes[row, 2].set_title("Clean Model Prediction")

        axes[row, 3].imshow(image)
        axes[row, 3].imshow(adv_pred_color, alpha=0.5)
        axes[row, 3].set_title(
            f"Adversarial Prediction ({num_steps}-step PGD, {num_restarts} restarts)"
        )

        axes[row, 4].imshow(diff_vis)
        axes[row, 4].set_title("Perturbation (Amplified 10x)")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def flatten_robustness_metrics(
    metrics: Dict[str, Union[float, List[float]]],
) -> Dict[str, float]:
    """Flatten clean/adversarial robustness metrics for CSV export.

    Steps
    -----
    1. Store scalar robustness metrics, including clean/adversarial cross-entropy,
       in one flat result row.
    2. Expand clean and adversarial per-class IoUs into named columns.

    Parameters
    ----------
    metrics : Dict[str, Union[float, List[float]]]
        Robustness metrics returned by :func:`evaluate_robustness`.

    Returns
    -------
    Dict[str, float]
        Flat row ready for DataFrame/CSV export.
    """
    row: Dict[str, float] = {
        "clean_mean_iou": cast(float, metrics["clean_mean_iou"]),
        "adv_mean_iou": cast(float, metrics["adv_mean_iou"]),
        "clean_pixel_accuracy": cast(float, metrics["clean_pixel_accuracy"]),
        "adv_pixel_accuracy": cast(float, metrics["adv_pixel_accuracy"]),
        "clean_dice": cast(float, metrics["clean_dice"]),
        "adv_dice": cast(float, metrics["adv_dice"]),
        "clean_cross_entropy": cast(float, metrics["clean_cross_entropy"]),
        "adv_cross_entropy": cast(float, metrics["adv_cross_entropy"]),
        "clean_batch_macro_iou": cast(float, metrics["clean_batch_macro_iou"]),
        "adv_batch_macro_iou": cast(float, metrics["adv_batch_macro_iou"]),
    }

    # Preserve class-wise clean/adversarial values and their relative loss so
    # the CSV contains the same robustness detail printed by the CLI report.
    clean_class_iou = cast(List[float], metrics["clean_class_iou"])
    adv_class_iou = cast(List[float], metrics["adv_class_iou"])
    for class_id in range(Configuration.NUM_VALID_CLASSES):
        class_slug = CLASS_NAMES[class_id].replace(" ", "_")
        clean_value = clean_class_iou[class_id]
        adv_value = adv_class_iou[class_id]

        row[f"clean_class_iou_{class_id}_{class_slug}"] = clean_value
        row[f"adv_class_iou_{class_id}_{class_slug}"] = adv_value

    return row


def parse_checkpoint_spec(value: str) -> Tuple[str, str]:
    """Parse a ``CHECKPOINT_NAME=CHECKPOINT_PATH`` command-line value.

    The checkpoint name becomes the result-directory name, while the path is
    passed unchanged to the model loader. Keeping both values in one named
    argument makes the output location deterministic for each checkpoint.

    Steps
    -----
    1. Split the input on the first ``=`` so checkpoint paths may contain
       additional ``=`` characters.
    2. Require both a non-empty checkpoint name and checkpoint path.
    3. Reject path separators in the checkpoint name because it must represent
       exactly one directory component under ``./model/robustness``.
    4. Return the validated ``(checkpoint_name, checkpoint_path)`` pair.

    Parameters
    ----------
    value : str
        Raw ``--checkpoint`` value in ``CHECKPOINT_NAME=CHECKPOINT_PATH`` form.

    Returns
    -------
    Tuple[str, str]
        Validated checkpoint name and checkpoint path.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not in ``NAME=PATH`` form, either side is empty, or the
        checkpoint name contains a path separator.
    """
    checkpoint_name, separator, checkpoint_path = value.partition("=")
    checkpoint_name = checkpoint_name.strip()
    checkpoint_path = checkpoint_path.strip()

    if separator == "" or not checkpoint_name or not checkpoint_path:
        raise argparse.ArgumentTypeError(
            "--checkpoint must use CHECKPOINT_NAME=CHECKPOINT_PATH, for example "
            "student_V5=./model/student_V5.pt."
        )

    # The name becomes one directory directly beneath the robustness result
    # root; rejecting separators prevents escaping or nesting that directory.
    if (
        checkpoint_name in {".", ".."}
        or "/" in checkpoint_name
        or "\\" in checkpoint_name
    ):
        raise argparse.ArgumentTypeError(
            "Checkpoint name must be a single directory name without path separators."
        )

    return checkpoint_name, checkpoint_path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the robustness evaluation script.

    Steps
    -----
    1. Require exactly one named ``--checkpoint`` value in ``NAME=PATH`` form.
    2. Add optional arguments for attack steps, random restarts, and visualization export.
    3. Parse and return the argument namespace.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate adversarial robustness of any trained segmentation model "
            "(.pt) using multi-restart PGD."
        )
    )
    # Keep the checkpoint name and path in one named argument so the name can
    # deterministically select the checkpoint-specific result directory.
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint_spec,
        required=True,
        metavar="NAME=PATH",
        help=(
            "Checkpoint name and trained model checkpoint path, for example "
            "student_V5=./model/student_V5.pt."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Maximum L-infinity perturbation radius. Defaults to 0.03.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Step size per PGD iteration. Defaults to 0.01.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of PGD iterations. Defaults to 10.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=5,
        help="Number of independent random PGD restarts. Defaults to 5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Configuration.BATCH_SIZE,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Export side-by-side adversarial comparison to ./model/adversarial_predictions.png.",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate model loading, data loading, and multi-restart PGD evaluation.

    Steps
    -----
    1. Parse checkpoint name/path and attack configuration from command-line arguments.
    2. Create the checkpoint-specific robustness result directory.
    3. Load trained model checkpoint and place on target compute device.
    4. Load test partition and construct evaluation DataLoader.
    5. Run clean and worst-of-restarts PGD adversarial evaluation.
    6. Save scalar and per-class robustness metrics to ``result.csv``.
    7. Log comprehensive metrics.
    8. Optionally export comparison visualizations.

    Raises
    ------
    FileNotFoundError
        If the requested checkpoint file does not exist.
    TypeError
        If the loaded checkpoint is not a ``torch.nn.Module``.
    ValueError
        If the evaluation dataloader yields zero batches.
    OSError
        If the checkpoint-specific result directory or CSV cannot be written.
    """
    args = parse_arguments()

    # Validate CLI values before loading data or model weights so configuration
    # mistakes fail quickly and cannot produce partially written result files.
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero.")
    if args.restarts <= 0:
        raise ValueError("--restarts must be greater than zero.")
    if args.epsilon < 0.0:
        raise ValueError("--epsilon must be non-negative.")
    if args.alpha <= 0.0:
        raise ValueError("--alpha must be greater than zero.")

    # Random starts are part of the robustness evaluation, so seed PyTorch to
    # keep repeated runs comparable while still using independent restart draws.
    torch.manual_seed(Configuration.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Configuration.SEED)

    print("\n" + "=" * 68)
    print("ROBUSTNESS EVALUATION PIPELINE (Multi-Restart PGD Attack)")
    print("=" * 68)

    checkpoint_name, checkpoint_path = args.checkpoint

    # Keep the CSV for each checkpoint isolated under the requested robustness
    # result root so repeated evaluations cannot overwrite another model's result.
    output_directory = os.path.join("./model", "robustness", checkpoint_name)
    os.makedirs(output_directory, exist_ok=True)

    # Resolve and load the checkpoint path only after the output location is
    # deterministically derived from its user-provided checkpoint name.
    print(f"Checkpoint name: {checkpoint_name}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = load_trained_model(checkpoint_path, Configuration.DEVICE)
    print(f"Target execution device: {Configuration.DEVICE}")

    # Load dataset test partition
    print("Loading BDD100k test split...")
    _, _, test_df = load_dataset_from_files()
    print(f"Loaded {len(test_df)} test samples for evaluation.")

    inference_transforms = A.Compose([
        A.Resize(height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH),
        ToTensorV2(),
    ])
    test_ds = BDDSegmentationDataset(test_df, transform=inference_transforms)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=False,  # Deterministic sample ordering for evaluation
        num_workers=Configuration.NUM_WORKERS,
    )

    print("\nRunning clean & adversarial evaluation:")
    print(f"  PGD Steps:   {args.steps}")
    print(f"  Restarts:    {args.restarts}")
    print(f"  Epsilon:     {args.epsilon}")
    print(f"  Alpha (LR):  {args.alpha}")
    print("-" * 68)

    results = evaluate_robustness(
        model=model,
        dataloader=test_loader,
        device=Configuration.DEVICE,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_steps=args.steps,
        num_restarts=args.restarts,
    )

    clean_iou = cast(float, results["clean_mean_iou"])
    adv_iou = cast(float, results["adv_mean_iou"])
    clean_acc = cast(float, results["clean_pixel_accuracy"])
    adv_acc = cast(float, results["adv_pixel_accuracy"])
    clean_dice = cast(float, results["clean_dice"])
    adv_dice = cast(float, results["adv_dice"])
    clean_cross_entropy = cast(float, results["clean_cross_entropy"])
    adv_cross_entropy = cast(float, results["adv_cross_entropy"])
    clean_class_iou = cast(List[float], results["clean_class_iou"])
    adv_class_iou = cast(List[float], results["adv_class_iou"])

    # Mirror the weather/time-of-day evaluator's flat CSV layout so all scalar
    # and class-wise robustness values remain machine-readable after evaluation.
    output_csv_path = os.path.join(output_directory, "result.csv")
    pd.DataFrame([flatten_robustness_metrics(results)]).to_csv(
        output_csv_path,
        index=False,
    )

    # Print summary report
    print("\n" + "=" * 68)
    print("EVALUATION RESULTS")
    print("=" * 68)
    print(f"  Clean Mean IoU:         {clean_iou:.4f}")
    print(f"  Adversarial Mean IoU:   {adv_iou:.4f}")
    print("-" * 68)
    print(f"  Clean Pixel Accuracy:   {clean_acc:.4f}")
    print(f"  Adv Pixel Accuracy:     {adv_acc:.4f}")
    print(f"  Clean Macro Dice:       {clean_dice:.4f}")
    print(f"  Adv Macro Dice:         {adv_dice:.4f}")
    print(f"  Clean Cross-Entropy:    {clean_cross_entropy:.6f}")
    print(f"  Adv Cross-Entropy:      {adv_cross_entropy:.6f}")
    print(f"  CSV Result:             {output_csv_path}")
    print("-" * 68)
    print("Per-Class IoU Breakdown:")
    print(f"  {'ID':>2}  {'Class Name':<15} {'Clean IoU':>10} {'Adv IoU':>10} {'Degradation':>12}")
    for cid in range(Configuration.NUM_VALID_CLASSES):
        c_name = CLASS_NAMES[cid]
        c_val = clean_class_iou[cid]
        a_val = adv_class_iou[cid]
        print(f"  {cid:>2}  {c_name:<15} {c_val:>10.4f} {a_val:>10.4f}")
    print("=" * 68 + "\n")

    if args.visualize:
        # Group each checkpoint's persisted metrics beneath its own directory so a
        # new evaluation cannot overwrite the result of a differently named model.
        output_predictions_path = os.path.join(output_directory, "adversarial_predictions.png")
        print(f"Exporting visual sample predictions to {output_predictions_path}...")
        visualize_adversarial_predictions(
            model=model,
            test_df=test_df,
            device=Configuration.DEVICE,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.steps,
            num_restarts=args.restarts,
            output_path=output_predictions_path,
        )
        print("Visualization saved successfully.")


if __name__ == "__main__":
    main()
