"""Robustness evaluation script for semantic segmentation models under PGD attack.

This module evaluates the adversarial robustness of trained PyTorch segmentation
models (.pt files) using a 10-step Projected Gradient Descent (PGD) untargeted attack.
It computes clean vs. adversarial segmentation metrics and calculates the untargeted
Attack Success Rate (ASR).
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


class ImagePath:
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
    return base_image_path + "/" + mask_file_name + ".jpg"


def find_train_image_path_from_mask(complete_mask_path: str) -> str:
    """Locate the training image corresponding to a training mask.

    Steps
    -----
    1. Delegate to ``find_image_path_from_mask`` using ``ImagePath.IMAGE_TRAIN_PATH``.

    Parameters
    ----------
    complete_mask_path : str
        Full path to a training mask PNG.

    Returns
    -------
    str
        Full path to the corresponding training image JPEG.
    """
    return find_image_path_from_mask(complete_mask_path, ImagePath.IMAGE_TRAIN_PATH)


def find_mask_path_from_image(complete_image_path: str, base_mask_path: str) -> str:
    """Derive the corresponding PNG mask path from a JPEG image path.

    Steps
    -----
    1. Extract the sample stem by stripping the ``.jpg`` extension.
    2. Append the ``_train_color.png`` suffix used by BDD100k masks.

    Parameters
    ----------
    complete_image_path : str
        Full filesystem path to an image JPEG.
    base_mask_path : str
        Directory containing the corresponding mask PNGs.

    Returns
    -------
    str
        Full filesystem path to the matching mask file.
    """
    file_path_split = complete_image_path.split("/")
    mask_file_name = file_path_split[-1].split(".")[0]
    return base_mask_path + "/" + mask_file_name + "_train_color.png"


def find_train_mask_path_from_image(complete_image_path: str) -> str:
    """Locate the training mask corresponding to a training image.

    Steps
    -----
    1. Delegate to ``find_mask_path_from_image`` using ``ImagePath.SEGMENTATION_MASK_TRAIN_PATH``.

    Parameters
    ----------
    complete_image_path : str
        Full path to a training image JPEG.

    Returns
    -------
    str
        Full path to the corresponding training mask PNG.
    """
    return find_mask_path_from_image(complete_image_path, ImagePath.SEGMENTATION_MASK_TRAIN_PATH)


def load_dataset_from_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan disk and load the BDD100k train, validation, and test splits.

    This function duplicates the exact partitioning logic from the training scripts
    so the evaluation evaluates the model on the identical test split.

    Steps
    -----
    1. Collect train masks and filter out corrupted or non-standard resolutions.
    2. Map to corresponding train images and filter any missing or malformed pairs.
    3. Partition the train-test pool using 80/20 train_test_split with fixed SEED.
    4. Collect and validate validation masks and images.
    5. Return DataFrames for train, validation, and test splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames representing ``(train_df, val_df, test_df)``.
    """
    train_mask_paths = glob.glob(f"{ImagePath.SEGMENTATION_MASK_TRAIN_PATH}/*.png")
    problematic_masks: List[str] = []
    for complete_mask_path in train_mask_paths:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_masks.append(complete_mask_path)
                train_mask_paths.remove(complete_mask_path)

    train_image_paths = list(map(find_train_image_path_from_mask, train_mask_paths))
    problematic_images: List[str] = []
    for complete_image_path in train_image_paths:
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_images.append(complete_image_path)
                train_image_paths.remove(complete_image_path)
                train_mask_paths.remove(find_train_mask_path_from_image(complete_image_path))

    train_test_df = pd.DataFrame({
        "image_paths": train_image_paths,
        "mask_paths": train_mask_paths,
    })
    train_df, test_df = train_test_split(
        train_test_df, test_size=0.2, random_state=Configuration.SEED
    )

    val_mask_paths = glob.glob(f"{ImagePath.SEGMENTATION_MASK_VAL_PATH}/*.png")
    for complete_mask_path in val_mask_paths:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                val_mask_paths.remove(complete_mask_path)

    val_image_paths = list(map(find_image_path_from_mask, val_mask_paths, [ImagePath.IMAGE_VAL_PATH] * len(val_mask_paths)))
    for complete_image_path in val_image_paths:
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                val_image_paths.remove(complete_image_path)
                val_mask_paths.remove(find_mask_path_from_image(complete_image_path, ImagePath.SEGMENTATION_MASK_VAL_PATH))

    val_df = pd.DataFrame({
        "image_paths": val_image_paths,
        "mask_paths": val_mask_paths,
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
def compute_batch_macro_iou(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int = 19,
    eps: float = 1e-7,
) -> float:
    """Compute per-batch macro IoU over non-empty classes.

    Steps
    -----
    1. Collapse predictions and targets to 2D class maps using argmax.
    2. Evaluate intersection and union class-by-class for classes 0 to ``num_classes - 1``.
    3. Average IoU strictly over classes that appear in either ground-truth or prediction.

    Parameters
    ----------
    y_pred : torch.Tensor
        Model output logits of shape ``(B, C, H, W)``.
    y_true : torch.Tensor
        One-hot ground-truth masks of shape ``(B, C, H, W)``.
    num_classes : int, optional
        Number of valid classes to consider. Defaults to 19.
    eps : float, optional
        Smoothing constant to prevent division by zero. Defaults to 1e-7.

    Returns
    -------
    float
        Macro IoU score averaged across present classes in the batch.
    """
    preds = y_pred.argmax(dim=1)
    targets = y_true.argmax(dim=1)

    iou_per_class: List[float] = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        true_mask = targets == cls

        intersection = (pred_mask & true_mask).sum().float().item()
        union = (pred_mask | true_mask).sum().float().item()

        if union > 0:
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


def resolve_model_path(model_arg: str) -> str:
    """Locate the requested checkpoint file on disk.

    This function makes the evaluation script reusable for any model name
    whether passed as a bare name (e.g. 'teacher'), filename ('teacher.pt'),
    or full path ('./model/teacher.pt').

    Steps
    -----
    1. Check if ``model_arg`` directly exists as a file.
    2. Search common fallback paths (appending ``.pt`` and checking the ``model/`` directory).
    3. Return the first matching path, or raise ``FileNotFoundError`` if unresolved.

    Parameters
    ----------
    model_arg : str
        Model name, filename, or filesystem path provided via command line.

    Returns
    -------
    str
        Resolved, valid filesystem path to the model checkpoint.

    Raises
    ------
    FileNotFoundError
        If no checkpoint file matching ``model_arg`` exists on disk.
    """
    candidates = [
        model_arg,
        f"{model_arg}.pt",
        os.path.join("model", model_arg),
        os.path.join("model", f"{model_arg}.pt"),
        os.path.join(".", "model", model_arg),
        os.path.join(".", "model", f"{model_arg}.pt"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate model checkpoint for '{model_arg}'. "
        f"Searched candidate paths:\n  - " + "\n  - ".join(candidates)
    )


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
    resolved_path = resolve_model_path(model_path)
    # weights_only=False is required because the project saves entire nn.Module objects
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, nn.Module):
        raise TypeError(
            f"Checkpoint at '{resolved_path}' loaded an object of type {type(checkpoint)}, "
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
    random_start: bool = False,
) -> BatchImage:
    """Generate untargeted adversarial perturbations using Projected Gradient Descent (PGD).

    PGD iteratively maximizes the segmentation loss with respect to the input image
    within an L-infinity ball of radius ``epsilon``. For untargeted attacks, maximizing
    cross-entropy loss pushes predictions away from the ground-truth annotations.

    Steps
    -----
    1. Freeze model in eval mode and prepare target class index map.
    2. Initialize perturbed input, optionally adding uniform random noise in [-epsilon, epsilon].
    3. For ``num_steps`` iterations:
       a. Track gradients on the current perturbed image.
       b. Compute model logits and evaluate segmentation loss.
       c. Compute input gradient via autograd and step in the direction of the sign.
       d. Project perturbation back onto the L-infinity epsilon ball.
       e. Clamp the resulting adversarial image to the valid [0, 1] range.
    4. Return the crafted adversarial image batch.

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
        Whether to initialize with random uniform perturbation. Defaults to False.

    Returns
    -------
    BatchImage
        Adversarial image batch of shape ``(B, 3, H, W)``.
    """
    model.eval()

    # Ignore class 19 (Unknown) so background noise does not misguide the attack
    if loss_fn is None:
        effective_loss: nn.Module = nn.CrossEntropyLoss(ignore_index=Configuration.IGNORED_CLASS)
    else:
        effective_loss = loss_fn

    # Collapse one-hot targets to class indices for standard cross-entropy evaluation
    target_indices = targets.argmax(dim=1).to(torch.int64)

    # Initialize perturbation
    if random_start:
        noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(images + noise, 0.0, 1.0).detach()
    else:
        x_adv = images.clone().detach()

    # Multi-step projected gradient ascent
    for _ in range(num_steps):
        # Enable autograd on input tensor to compute dLoss/dx_adv
        x_adv = x_adv.clone().detach().requires_grad_(True)

        adv_logits = forward(model, x_adv)
        loss = effective_loss(adv_logits, target_indices)

        # Compute gradient with respect to perturbed input
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        # Step along gradient sign (gradient ascent), project onto L_inf ball, and clamp to [0, 1]
        perturbation = x_adv.detach() + alpha * grad.sign() - images
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        x_adv = torch.clamp(images + perturbation, 0.0, 1.0).detach()

    return cast(BatchImage, x_adv)


def compute_untargeted_asr(clean_mean_iou: float, adv_mean_iou: float) -> float:
    """Calculate the untargeted Attack Success Rate (ASR) from mean IoU values.

    The formula measures relative degradation in mean IoU caused by the adversary:
    ``ASR_untargeted = 1 - (mean_IoU_adversarial / mean_IoU_clean)``.
    A score of 1.0 represents complete destruction of segmentation utility,
    while 0.0 indicates absolute robustness.

    Steps
    -----
    1. Guard against non-positive clean mean IoU to avoid division by zero.
    2. Compute and return 1.0 - (adv_mean_iou / clean_mean_iou).

    Parameters
    ----------
    clean_mean_iou : float
        Mean Intersection over Union on clean, unperturbed inputs.
    adv_mean_iou : float
        Mean Intersection over Union on adversarial inputs.

    Returns
    -------
    float
        Untargeted ASR value.
    """
    # Guard against division by zero if clean model has zero performance
    if clean_mean_iou <= 0.0:
        return 0.0
    return float(1.0 - (adv_mean_iou / clean_mean_iou))


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
) -> Dict[str, Union[float, List[float]]]:
    """Run full clean and 10-step PGD adversarial evaluation on a dataloader.

    Confusion counts (TP, FP, FN, TN) are accumulated over the entire test set
    before reduction, guaranteeing statistically robust metrics regardless of
    batch-level class imbalances.

    Steps
    -----
    1. Iterate through dataloader batches.
    2. Run clean forward pass under ``torch.no_grad`` to obtain clean logits.
    3. Generate 10-step PGD adversarial images using ``pgd_attack``.
    4. Run adversarial forward pass under ``torch.no_grad`` to obtain perturbed logits.
    5. Accumulate per-class confusion statistics for both clean and adversarial outputs.
    6. Compute dataset-wide macro IoU, pixel accuracy, Dice score, and per-class IoU.
    7. Calculate untargeted ASR via ``compute_untargeted_asr``.

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

    Returns
    -------
    Dict[str, Union[float, List[float]]]
        Dictionary with clean/adversarial metrics and untargeted ASR.

    Raises
    ------
    ValueError
        If the dataloader produces zero batches.
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

    for X, y in tqdm(dataloader, desc="Evaluating robustness"):
        X, y = X.to(device), y.to(device)

        # 1. Clean forward pass (no gradients needed)
        with torch.no_grad():
            clean_logits = forward(model, X)
            clean_preds = clean_logits.argmax(dim=1)

        # 2. Craft 10-step PGD adversarial examples
        adv_X = pgd_attack(
            model=model,
            images=X,
            targets=y,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
        )

        # 3. Adversarial forward pass
        with torch.no_grad():
            adv_logits = forward(model, adv_X)
            adv_preds = adv_logits.argmax(dim=1)

        # 4. Relabel Unknown class (19) to sentinel -1 so smp excludes it from metrics
        targets = y.argmax(dim=1)
        target_relabelled = torch.where(
            targets == ignored_class, torch.full_like(targets, ignore_index), targets
        )
        clean_preds_relabelled = torch.where(
            clean_preds == ignored_class, torch.full_like(clean_preds, ignore_index), clean_preds
        )
        adv_preds_relabelled = torch.where(
            adv_preds == ignored_class, torch.full_like(adv_preds, ignore_index), adv_preds
        )

        # 5. Accumulate clean confusion counts
        c_tp, c_fp, c_fn, c_tn = smp.metrics.functional.get_stats(
            clean_preds_relabelled,
            target_relabelled,
            mode="multiclass",
            ignore_index=ignore_index,
            num_classes=num_classes,
        )
        clean_tp_total = c_tp.sum(dim=0) if clean_tp_total is None else clean_tp_total + c_tp.sum(dim=0)
        clean_fp_total = c_fp.sum(dim=0) if clean_fp_total is None else clean_fp_total + c_fp.sum(dim=0)
        clean_fn_total = c_fn.sum(dim=0) if clean_fn_total is None else clean_fn_total + c_fn.sum(dim=0)
        clean_tn_total = c_tn.sum(dim=0) if clean_tn_total is None else clean_tn_total + c_tn.sum(dim=0)

        # 6. Accumulate adversarial confusion counts
        a_tp, a_fp, a_fn, a_tn = smp.metrics.functional.get_stats(
            adv_preds_relabelled,
            target_relabelled,
            mode="multiclass",
            ignore_index=ignore_index,
            num_classes=num_classes,
        )
        adv_tp_total = a_tp.sum(dim=0) if adv_tp_total is None else adv_tp_total + a_tp.sum(dim=0)
        adv_fp_total = a_fp.sum(dim=0) if adv_fp_total is None else adv_fp_total + a_fp.sum(dim=0)
        adv_fn_total = a_fn.sum(dim=0) if adv_fn_total is None else adv_fn_total + a_fn.sum(dim=0)
        adv_tn_total = a_tn.sum(dim=0) if adv_tn_total is None else adv_tn_total + a_tn.sum(dim=0)

        # Per-batch macro IoU tracking
        clean_batch_iou_sum += compute_batch_macro_iou(clean_logits, y, num_classes=num_classes)
        adv_batch_iou_sum += compute_batch_macro_iou(adv_logits, y, num_classes=num_classes)

    if clean_tp_total is None or adv_tp_total is None:
        raise ValueError("The evaluation dataloader yielded zero batches.")

    # Macro mean IoU across the whole dataset (primary metric for ASR)
    clean_mean_iou = float(smp.metrics.iou_score(clean_tp_total, clean_fp_total, clean_fn_total, clean_tn_total, reduction="macro").item())
    adv_mean_iou = float(smp.metrics.iou_score(adv_tp_total, adv_fp_total, adv_fn_total, adv_tn_total, reduction="macro").item())

    # Calculate untargeted ASR
    untargeted_asr = compute_untargeted_asr(clean_mean_iou=clean_mean_iou, adv_mean_iou=adv_mean_iou)

    # Class-wise IoUs
    clean_class_iou = cast(List[float], smp.metrics.iou_score(clean_tp_total, clean_fp_total, clean_fn_total, clean_tn_total, reduction=None).tolist())
    adv_class_iou = cast(List[float], smp.metrics.iou_score(adv_tp_total, adv_fp_total, adv_fn_total, adv_tn_total, reduction=None).tolist())

    # Pixel accuracy and Dice (F1) scores
    clean_pixel_acc = float(smp.metrics.accuracy(clean_tp_total, clean_fp_total, clean_fn_total, clean_tn_total, reduction="micro").item())
    adv_pixel_acc = float(smp.metrics.accuracy(adv_tp_total, adv_fp_total, adv_fn_total, adv_tn_total, reduction="micro").item())
    clean_dice = float(smp.metrics.f1_score(clean_tp_total, clean_fp_total, clean_fn_total, clean_tn_total, reduction="macro").item())
    adv_dice = float(smp.metrics.f1_score(adv_tp_total, adv_fp_total, adv_fn_total, adv_tn_total, reduction="macro").item())

    num_batches = len(dataloader)
    return {
        "clean_mean_iou": clean_mean_iou,
        "adv_mean_iou": adv_mean_iou,
        "untargeted_asr": untargeted_asr,
        "clean_pixel_accuracy": clean_pixel_acc,
        "adv_pixel_accuracy": adv_pixel_acc,
        "clean_dice": clean_dice,
        "adv_dice": adv_dice,
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
    output_path: str = "./adversarial_predictions.png",
) -> None:
    """Render a visual side-by-side comparison of clean and adversarial predictions.

    Steps
    -----
    1. Sample random test rows using the fixed seed.
    2. Craft adversarial examples for each sample.
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
        Destination image path. Defaults to "./adversarial_predictions.png".

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
        axes[row, 3].set_title("Adversarial Prediction (10-step PGD)")

        axes[row, 4].imshow(diff_vis)
        axes[row, 4].set_title("Perturbation (Amplified 10x)")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the robustness evaluation script.

    Steps
    -----
    1. Configure ArgumentParser with mandatory model checkpoint positional argument.
    2. Add optional arguments for attack parameters and visualization export.
    3. Parse and return the argument namespace.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial robustness of any trained segmentation model (.pt) using 10-step PGD."
    )
    # The 1 mandatory argument requested by the user
    parser.add_argument(
        "model_path",
        type=str,
        help="Filename or path of the trained model checkpoint (.pt file).",
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
        "--batch-size",
        type=int,
        default=Configuration.BATCH_SIZE,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export side-by-side adversarial comparison to ./adversarial_predictions.png.",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate model loading, test data loading, and 10-step PGD robustness evaluation.

    Steps
    -----
    1. Parse model path and attack configuration from command-line arguments.
    2. Load trained model checkpoint and place on target compute device.
    3. Load test partition and construct evaluation DataLoader.
    4. Run clean and 10-step PGD adversarial evaluation.
    5. Log comprehensive metrics and untargeted Attack Success Rate (ASR).
    6. Optionally export comparison visualizations.
    """
    args = parse_arguments()

    print("\n" + "=" * 68)
    print("ROBUSTNESS EVALUATION PIPELINE (10-Step PGD Attack)")
    print("=" * 68)

    # Resolve and load model checkpoint
    resolved_path = resolve_model_path(args.model_path)
    print(f"Loading checkpoint from: {resolved_path}")
    model = load_trained_model(resolved_path, Configuration.DEVICE)
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
    )

    clean_iou = cast(float, results["clean_mean_iou"])
    adv_iou = cast(float, results["adv_mean_iou"])
    asr = cast(float, results["untargeted_asr"])
    clean_acc = cast(float, results["clean_pixel_accuracy"])
    adv_acc = cast(float, results["adv_pixel_accuracy"])
    clean_dice = cast(float, results["clean_dice"])
    adv_dice = cast(float, results["adv_dice"])
    clean_class_iou = cast(List[float], results["clean_class_iou"])
    adv_class_iou = cast(List[float], results["adv_class_iou"])

    # Print summary report
    print("\n" + "=" * 68)
    print("EVALUATION RESULTS & ATTACK SUCCESS RATE")
    print("=" * 68)
    print(f"  Clean Mean IoU:         {clean_iou:.4f}")
    print(f"  Adversarial Mean IoU:   {adv_iou:.4f}")
    print(f"  Untargeted ASR:         {asr:.4f} ({asr * 100.0:.2f}%)")
    print("-" * 68)
    print(f"  Clean Pixel Accuracy:   {clean_acc:.4f}")
    print(f"  Adv Pixel Accuracy:     {adv_acc:.4f}")
    print(f"  Clean Macro Dice:       {clean_dice:.4f}")
    print(f"  Adv Macro Dice:         {adv_dice:.4f}")
    print("-" * 68)
    print("Per-Class IoU Breakdown:")
    print(f"  {'ID':>2}  {'Class Name':<15} {'Clean IoU':>10} {'Adv IoU':>10} {'Degradation':>12}")
    for cid in range(Configuration.NUM_VALID_CLASSES):
        c_name = CLASS_NAMES[cid]
        c_val = clean_class_iou[cid]
        a_val = adv_class_iou[cid]
        deg = ((c_val - a_val) / c_val * 100.0) if c_val > 0 else 0.0
        print(f"  {cid:>2}  {c_name:<15} {c_val:>10.4f} {a_val:>10.4f} {deg:>11.2f}%")
    print("=" * 68 + "\n")

    if args.visualize:
        print("Exporting visual sample predictions to ./adversarial_predictions.png...")
        visualize_adversarial_predictions(
            model=model,
            test_df=test_df,
            device=Configuration.DEVICE,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.steps,
            output_path="./adversarial_predictions.png",
        )
        print("Visualization saved successfully.")


if __name__ == "__main__":
    main()
