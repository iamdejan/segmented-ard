"""Evaluate segmentation performance by BDD100k weather/time-of-day pair.

This module adjusts ``evaluate_performance.py`` so evaluation is driven by
``./data/weather_timeofday_pair_data.csv`` and the image-level metadata in
``./data/bdd100k_10k_images_with_100k_metadata_intersection.csv``. The original
BDD100k held-out segmentation split and metric logic are preserved. The
intersection CSV supplies concrete image IDs and their weather/time-of-day
labels. ``analyze_100k_10k_intersection.py`` generates the pair CSV from this
same held-out segmentation population, so every pair count is required to equal
the number of samples actually evaluated for that condition.
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple, cast

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# Shape aliases documented alongside each pipeline stage. ``jaxtyped`` +
# ``beartype`` enforce them at runtime so shape mismatches surface as clear
# ``TypeCheckError`` exceptions instead of confusing downstream broadcasts.
ImageTensor = Float[Tensor, "3 h w"]  # single image, channel-first layout
MaskTensor = Float[Tensor, "c h w"]  # one-hot mask, c == NUM_CLASSES
BatchImage = Float[Tensor, "b 3 h w"]  # collated batch of images
BatchMask = Float[Tensor, "b c h w"]  # collated batch of one-hot masks
Logits = Float[Tensor, "b c h w"]  # model output, c == NUM_CLASSES
NumpyImage = Float[np.ndarray, "h w 3"]  # single image, channels-last layout
ClassIndexArray = UInt8[np.ndarray, "h w"]  # per-pixel class id map (numpy)


# BDD100k color-label palette. The row index is the class id (0-19), matching
# ``Configuration.NUM_CLASSES``. Predicted class maps are coloured with this
# same palette so they render side by side with the ground-truth color labels
# stored on disk. The exact colour per class only needs to be distinct and
# consistent; it mirrors the default BDD100k colours.
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


# Human-readable names for each palette row, kept in lock-step with
# ``CLASS_COLORS`` so class ids can be logged without consulting the palette
# comments by hand.
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
    NUM_WORKERS = 2

    BATCH_SIZE = (
        16 if torch.cuda.device_count() < 2
        else (16 * torch.cuda.device_count())
    )
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

    WEATHER_TIMEOFDAY_PAIR_CSV = "./data/weather_timeofday_pair_data.csv"
    METADATA_INTERSECTION_CSV = (
        "./data/bdd100k_10k_images_with_100k_metadata_intersection.csv"
    )


def color_label_to_class_index(label: np.ndarray) -> np.ndarray:
    """Map an RGB color-label image to a per-pixel class-index map.

    BDD100k stores segmentation masks as RGB PNGs whose colours are exactly
    the entries of ``CLASS_COLORS``. Semantic segmentation needs the class id
    per pixel (shape ``(H, W)``) rather than the RGB representation (shape
    ``(H, W, 3)``), so this conversion must happen before the mask is turned
    into a tensor and one-hot encoded.

    Steps
    -----
    1. Initialise the output with the id of the last palette entry so that any
       unknown colour degrades to ``Unknown`` instead of producing an invalid
       index.
    2. For each palette colour, boolean-mask the pixels whose RGB values match
       it exactly and assign the corresponding class id. The loop is over only
       ``NUM_CLASSES`` colours and each iteration is fully vectorised.

    Parameters
    ----------
    label : np.ndarray
        RGB color-label array of shape ``(H, W, 3)`` with integer values.

    Returns
    -------
    np.ndarray
        Class-index array of shape ``(H, W)`` and dtype ``uint8``, whose values
        are in ``[0, NUM_CLASSES)``.
    """
    # Default to the last class id so unknown colours fall back gracefully
    # instead of indexing the palette out of bounds later.
    class_ids = np.full(label.shape[:2], CLASS_COLORS.shape[0] - 1, dtype=np.uint8)

    # Match each palette colour via exact RGB equality. This stays fast because
    # every comparison operates on the whole image at once.
    for class_id, (red, green, blue) in enumerate(CLASS_COLORS):
        match = (
            (label[..., 0] == red)
            & (label[..., 1] == green)
            & (label[..., 2] == blue)
        )
        class_ids[match] = class_id

    return class_ids


class BDDSegmentationDataset(Dataset[Tuple[ImageTensor, MaskTensor]]):
    """BDD100k semantic segmentation dataset loader.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``image_paths`` and ``mask_paths`` columns.
    transform : A.Compose | None, optional
        Albumentations transform pipeline applied to image and mask. Defaults
        to ``None``.
    """

    def __init__(self, df: pd.DataFrame, transform: A.Compose | None = None) -> None:
        super().__init__()

        self.image_paths: List[str] = df["image_paths"].to_list()
        self.mask_paths: List[str] = df["mask_paths"].to_list()
        self.transform = transform

    @jaxtyped(typechecker=beartype)
    def load_sample(self, index: int) -> Tuple[NumpyImage, ClassIndexArray]:
        """Load the image and its per-pixel class-index mask at ``index``.

        The image is opened as RGB and normalised to ``[0, 1]``. The mask PNG
        is likewise forced to RGB (some BDD100k color-labels carry an alpha
        channel) before being collapsed from the RGB color-label format to a
        single class id per pixel via :func:`color_label_to_class_index`. This
        class-index representation is what the one-hot encoder downstream
        expects.

        Steps
        -----
        1. Open both files as RGB and convert them to NumPy arrays.
        2. Normalise the image pixels to ``[0, 1]``.
        3. Map the mask's RGB colours to class indices.

        Parameters
        ----------
        index : int
            Zero-based position of the sample to load.

        Returns
        -------
        Tuple[NumpyImage, ClassIndexArray]
            The ``(image, mask)`` pair where ``image`` has shape ``(H, W, 3)``
            with float32 values in ``[0, 1]`` and ``mask`` has shape ``(H, W)``
            with uint8 class ids.

        Raises
        ------
        IndexError
            If ``index`` is out of the range of the dataset lists.
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Force RGB so that RGBA sources (e.g. some BDD100k color-label PNGs
        # carry an alpha channel) are reduced to 3 channels.
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
        """Return the transformed ``(image, mask)`` pair at position ``index``.

        ``ToTensorV2`` converts the image from ``(H, W, 3)`` to ``(3, H, W)``
        and leaves the 2-D class-index mask as ``(H, W)``. The mask is then
        one-hot encoded to ``(NUM_CLASSES, H, W)`` so that its channel axis
        lines up with the model logits. The DataLoader adds the batch dimension
        when collating samples.

        Parameters
        ----------
        index : int
            Zero-based position of the sample to load.

        Returns
        -------
        Tuple[ImageTensor, MaskTensor]
            The ``(image, mask)`` pair where ``image`` has shape ``(3, H, W)``
            and ``mask`` is a one-hot tensor of shape ``(NUM_CLASSES, H, W)``.
        """
        image, class_mask = self.load_sample(index)

        # Transform if necessary. The mask is a 2-D class map, so no channel
        # transposition is needed for it.
        if self.transform:
            transformed = self.transform(image=image, mask=class_mask)
        else:
            transformed = ToTensorV2()(image=image, mask=class_mask)

        # One-hot encode the (H, W) class ids into (NUM_CLASSES, H, W) floats
        # so the mask matches the model's (B, NUM_CLASSES, H, W) output.
        # ``one_hot`` needs int64 input, hence the cast.
        mask_one_hot = torch.nn.functional.one_hot(
            transformed["mask"].to(torch.int64), Configuration.NUM_CLASSES
        ).permute(2, 0, 1).float()

        return transformed["image"], mask_one_hot


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

    mask_path = base_mask_path + "/" + mask_file_name + "_train_color.png"
    return mask_path


def find_val_mask_path_from_image(complete_image_path: str) -> str:
    """Locate the validation mask corresponding to a validation image.

    Steps
    -----
    1. Delegate to :func:`find_mask_path_from_image` using
       ``Path.SEGMENTATION_MASK_VAL_PATH``.

    Parameters
    ----------
    complete_image_path : str
        Full path to a validation image JPEG.

    Returns
    -------
    str
        Full path to the corresponding validation mask PNG.
    """
    return find_mask_path_from_image(complete_image_path, Path.SEGMENTATION_MASK_VAL_PATH)


def load_dataset_from_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan disk and load the BDD100k train, validation, and test splits.

    This function duplicates the exact partitioning logic from the training
    scripts so the model is evaluated on the identical test split that was held
    out during training.

    Steps
    -----
    1. Collect train masks and filter out corrupted or non-standard resolutions.
    2. Map to corresponding train images and filter any missing or malformed pairs.
    3. Partition the train-test pool using 80/20 ``train_test_split`` with the
       fixed ``Configuration.SEED``.
    4. Collect and validate validation masks and images.
    5. Return DataFrames for train, validation, and test splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames representing ``(train_df, val_df, test_df)``.
    """
    # Build the valid training pool without mutating a list during iteration.
    # Preserving candidate order matters because the seeded train/test split is
    # deterministic only when it receives the same ordered population.
    train_mask_candidates = glob.glob(
        f"{Path.SEGMENTATION_MASK_TRAIN_PATH}/*.png"
    )
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

    # Load the BDD100k validation split, again filtering non-standard samples.
    val_mask_paths = glob.glob(f"{Path.SEGMENTATION_MASK_VAL_PATH}/*.png")
    problematic_val_masks: List[str] = []
    for complete_mask_path in val_mask_paths:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_val_masks.append(complete_mask_path)
                val_mask_paths.remove(complete_mask_path)
    print(f"Problematic val masks: {problematic_val_masks}")

    val_image_paths = list(map(find_val_image_path_from_mask, val_mask_paths))
    problematic_val_images: List[str] = []
    for complete_image_path in val_image_paths:
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_val_images.append(complete_image_path)
                val_image_paths.remove(complete_image_path)
                val_mask_paths.remove(find_val_mask_path_from_image(complete_image_path))
    print(f"Problematic val images: {problematic_val_images}")

    val_df = pd.DataFrame({
        "image_paths": val_image_paths,
        "mask_paths": val_mask_paths,
    })

    return train_df, val_df, test_df


@jaxtyped(typechecker=beartype)
def forward(model: nn.Module, x: BatchImage) -> Logits:
    """Run a single forward pass and assert the input/output shapes.

    Centralising the forward pass here lets ``jaxtyping`` verify that the input
    batch is always ``(B, 3, H, W)`` and that the model produces
    ``(B, NUM_CLASSES, H, W)`` logits, which is where shape confusion most often
    arises.

    Parameters
    ----------
    model : nn.Module
        The segmentation model to run.
    x : BatchImage
        Input batch of images with shape ``(B, 3, H, W)``.

    Returns
    -------
    Logits
        Raw model logits with shape ``(B, NUM_CLASSES, H, W)``.
    """
    return cast(Logits, model(x))


def evaluate_segmentation_metrics(
    model: nn.Module,
    dataloader: DataLoader[Tuple[ImageTensor, MaskTensor]],
    device: torch.device,
    num_classes: int = 19,
    ignore_index: int = -1,
    ignored_class: int = 19,
) -> Dict[str, float | List[float]]:
    """Compute pixel accuracy, per-class accuracy, IoU and Dice over a dataset.

    The metrics are derived from per-class true/false positive/negative pixel
    counts accumulated across *every* batch via :func:`smp.metrics.get_stats`,
    then reduced once at the end. Accumulating the raw counts rather than
    averaging per-batch values guarantees the metrics stay correct even when
    batches are class-imbalanced. The ``smp`` implementations are reused
    wherever possible:

    * Pixel accuracy averages every pixel, so it uses ``reduction="micro"`` and
      yields a single scalar.
    * Class-wise pixel accuracy scores each class independently, so it uses no
      reduction and returns one accuracy value per class.
    * Class-wise IoU (Jaccard) similarly uses no reduction and returns one IoU
      value per class; the mean IoU is then the average of those per-class
      values, matching macro-averaging.
    * Dice (F1) is macro-averaged over classes into a single scalar.

    The ``Unknown`` class (id ``ignored_class``) is remapped to the sentinel
    ``ignore_index`` so those pixels never influence any metric. This mirrors
    the ``ignore_index=19`` used by the training losses, but is required because
    :func:`smp.metrics.get_stats` only accepts an ``ignore_index`` that lies
    *outside* the ``[0, num_classes)`` range.

    Steps
    -----
    1. Run the model over every batch under ``torch.inference_mode``.
    2. Collapse logits and one-hot targets to ``(B, H, W)`` class-index maps and
       remap ``ignored_class`` to ``ignore_index`` on both.
    3. Accumulate per-class ``(tp, fp, fn, tn)`` pixel counts across batches.
    4. Reduce the accumulated counts into the requested metrics.

    Parameters
    ----------
    model : nn.Module
        Segmentation model to evaluate.
    dataloader : DataLoader[Tuple[ImageTensor, MaskTensor]]
        Batched data to evaluate over (typically the test set).
    device : torch.device
        Device the evaluation runs on.
    num_classes : int, optional
        Number of classes to score, excluding the ignored class. Defaults to
        ``19``.
    ignore_index : int, optional
        Sentinel class id excluded from every metric. Defaults to ``-1``.
    ignored_class : int, optional
        Original class id to exclude (``Unknown``). Defaults to ``19``.

    Returns
    -------
    Dict[str, float | List[float]]
        Mapping of metric name to value. ``pixel_accuracy``, ``iou`` and
        ``dice`` are scalars, while ``class_pixel_accuracy`` and ``class_iou``
        are lists whose index ``i`` holds the value of class ``i`` (classes
        ``0..18``).
    """
    model.eval()

    # Per-class confusion counts summed over the whole dataset. ``get_stats``
    # returns ``(N, C)`` tensors, so summing over the batch axis (dim 0) yields
    # one count per class.
    tp_total: Tensor | None = None
    fp_total: Tensor | None = None
    fn_total: Tensor | None = None
    tn_total: Tensor | None = None

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Feed-forward and reduce model logits to one class id per pixel.
            y_pred = forward(model, X)
            output = y_pred.argmax(dim=1)  # (B, H, W)
            target = y.argmax(dim=1)  # (B, H, W)

            # ``smp`` requires ``ignore_index`` outside the valid class range,
            # so the ``Unknown`` class (id 19) is relabelled to ``-1``.
            output = torch.where(
                output == ignored_class, torch.full_like(output, ignore_index), output
            )
            target = torch.where(
                target == ignored_class, torch.full_like(target, ignore_index), target
            )

            tp, fp, fn, tn = smp.metrics.functional.get_stats(
                output,
                target,
                mode="multiclass",
                ignore_index=ignore_index,
                num_classes=num_classes,
            )

            # Accumulate the ``(N, C)`` counts into per-class running sums.
            tp_total = tp.sum(dim=0) if tp_total is None else tp_total + tp.sum(dim=0)
            fp_total = fp.sum(dim=0) if fp_total is None else fp_total + fp.sum(dim=0)
            fn_total = fn.sum(dim=0) if fn_total is None else fn_total + fn.sum(dim=0)
            tn_total = tn.sum(dim=0) if tn_total is None else tn_total + tn.sum(dim=0)

    # Guard against an empty dataloader producing no counts at all.
    if tp_total is None:
        raise ValueError("The evaluation dataloader yielded no batches.")

    # Pixel accuracy: ratio of correctly classified pixels over all pixels.
    pixel_accuracy = tp_total.sum() / (tp_total + fn_total).sum()

    # Class-wise pixel accuracy: one accuracy value per class (no reduction).
    # ``reduction=None`` returns a ``(num_classes,)`` tensor whose element ``i``
    # is the accuracy of class ``i``.
    class_pixel_accuracy = tp_total / (tp_total + fn_total).clamp_min(1)

    # IoU (Jaccard index) and Dice (F1) macro-averaged over classes.
    iou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()
    dice = smp.metrics.f1_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()

    # Class-wise IoU: one IoU value per class (no reduction). ``reduction=None``
    # returns a ``(num_classes,)`` tensor whose element ``i`` is the IoU of
    # class ``i``.
    class_iou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction=None).tolist()

    # Mean IoU: average of the class-wise IoU values (macro-averaging).
    mean_iou = np.mean(class_iou)

    # Dataset-level global (micro) IoU
    global_iou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()

    return {
        "pixel_accuracy": float(pixel_accuracy),
        # Convert the per-class tensor to host Python floats here so callers can
        # safely average repeated runs even when evaluation executes on CUDA.
        "class_pixel_accuracy": class_pixel_accuracy.tolist(),
        "iou": float(iou),
        "mean_iou": float(mean_iou),
        "global_iou": float(global_iou),
        "class_iou": list(class_iou),
        "dice": float(dice),
    }


def load_trained_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a trained segmentation model checkpoint from disk.

    Steps
    -----
    1. Load the checkpoint using ``torch.load`` with ``weights_only=False`` to
       unpack serialized ``nn.Module`` objects produced by the training scripts.
    2. Transfer the model to the target ``device`` and switch into ``eval()``
       mode.

    Parameters
    ----------
    checkpoint_path : str
        Filesystem path to the model ``.pt`` file.
    device : torch.device
        Device (CUDA or CPU) on which to place the model.

    Returns
    -------
    nn.Module
        Loaded PyTorch segmentation model ready for evaluation.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` does not exist.
    TypeError
        If the loaded checkpoint object is not an ``nn.Module``.
    """
    # weights_only=False is required because the training scripts save entire
    # nn.Module objects rather than raw state dicts.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, nn.Module):
        raise TypeError(
            f"Checkpoint at '{checkpoint_path}' loaded an object of type {type(checkpoint)}, "
            "but an nn.Module instance was expected."
        )

    model = checkpoint.to(device)
    model.eval()

    return model


def colorize_mask(class_mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map a class-index mask to an RGB image using ``palette``.

    Steps
    -----
    1. Cast ``class_mask`` to integer so it can be used as row indices.
    2. Index ``palette`` with those indices, turning a ``(H, W)`` array of
       class ids into a ``(H, W, 3)`` image.

    Parameters
    ----------
    class_mask : np.ndarray
        Array of shape ``(H, W)`` whose values are class indices.
    palette : np.ndarray
        Array of shape ``(NUM_CLASSES, 3)`` mapping a class id to an RGB
        colour (the 0-255 range).

    Returns
    -------
    np.ndarray
        RGB image of shape ``(H, W, 3)`` matching the dtype of ``palette``.
    """
    return palette[class_mask.astype(np.int64)]


def visualize_predictions(
    model: nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    num_samples: int = 4,
    output_path: str = "./model/predictions.png",
) -> None:
    """Render a fixed set of test samples next to their true and predicted masks.

    The first ``num_samples`` rows of ``test_df`` are selected, every image is
    run through ``model``, and a grid with three columns (image / image + true
    mask / image + predicted mask) is exported to a PNG file. The model's
    ``(20, H, W)`` logits are reduced to a single class id per pixel via
    ``argmax`` so they can be coloured with ``CLASS_COLORS`` and compared to
    the ground-truth color labels.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model returning ``(B, 20, H, W)`` logits.
    test_df : pd.DataFrame
        DataFrame carrying the ``image_paths`` and ``mask_paths`` columns.
    device : torch.device
        Device used to run inference.
    num_samples : int, optional
        Number of samples to visualise. Defaults to 4.
    output_path : str, optional
        Destination of the exported PNG. Defaults to
        ``"./model/predictions.png"``.

    Raises
    ------
    ValueError
        If ``test_df`` has no rows to visualise.
    """
    # Take the first ``num_samples`` rows of the test frame (or fewer if the
    # frame is smaller) so the visualisation is identical across training runs.
    sample_df = test_df.head(min(num_samples, len(test_df))).reset_index(drop=True)

    if sample_df.empty:
        raise ValueError("test_df has no rows to visualise.")

    num_rows = len(sample_df)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # plt.subplots returns a 1D array when there is a single row; promote it
    # to 2D so the axes[row, col] indexing below is uniform.
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    # Reuse the dataset loader so the visual path matches training: this
    # guarantees the RGB collapse and [0, 1] scaling are identical.
    sample_ds = BDDSegmentationDataset(sample_df)

    # Switch to inference once for the whole grid; no gradients are needed.
    # Use the underlying unwrapped module for batch_size=1 inference
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.eval()

    for row in range(num_rows):
        # Load the raw pair: the image as an (H, W, 3) float array in [0, 1]
        # and the mask as an (H, W) class-index array.
        image, class_mask = sample_ds.load_sample(row)

        # Replicate the ToTensorV2 conversion: transpose HWC -> CHW, add a
        # batch dim and move to the device so the model sees the same format
        # it received during training.
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).contiguous()
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Forward pass, then collapse the 20-class logits to one class id per
        # pixel so the output can be colourised.
        with torch.inference_mode():
            logits = eval_model(image_tensor)
        pred_class = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Colour the predicted class map, then bring it back to [0, 1] for
        # Matplotlib so it can be blended with the RGB image.
        pred_color = colorize_mask(pred_class, CLASS_COLORS).astype(np.float32) / 255.0

        # Colour the ground-truth class map the same way so the two overlays
        # are directly comparable.
        true_mask_color = colorize_mask(class_mask, CLASS_COLORS).astype(np.float32) / 255.0

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Image")

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(true_mask_color, alpha=0.5)
        axes[row, 1].set_title("Image + True Mask")

        axes[row, 2].imshow(image)
        axes[row, 2].imshow(pred_color, alpha=0.5)
        axes[row, 2].set_title("Image + Predicted Mask")

        # Remove axis ticks/labels so only the pixels are shown.
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # Ensure the output directory exists before writing the PNG.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_pair_specification(pair_csv_path: str) -> pd.DataFrame:
    """Load and validate the requested weather/time-of-day pair specification.

    The ``count`` column is the expected number of samples in the evaluator's
    held-out segmentation subset after joining to the 10K/100K metadata
    intersection and removing undefined environment labels. It must therefore
    equal the number of samples actually evaluated for each pair.

    Steps
    -----
    1. Read the CSV and discard any exported unnamed index column.
    2. Require ``weather``, ``time_of_day``, and ``count`` columns.
    3. Normalize surrounding whitespace and validate unique, non-negative counts.

    Parameters
    ----------
    pair_csv_path : str
        Path to the weather/time-of-day pair CSV.

    Returns
    -------
    pd.DataFrame
        Validated pair table with ``weather``, ``time_of_day``, and ``count``.

    Raises
    ------
    FileNotFoundError
        If ``pair_csv_path`` does not exist.
    ValueError
        If required columns or valid pair/count values are missing.
    """
    # Validate the specification before doing model work so bad inputs fail early.
    if not os.path.isfile(pair_csv_path):
        raise FileNotFoundError(f"Pair CSV not found: {pair_csv_path}")

    pair_df = pd.read_csv(pair_csv_path)
    pair_df = pair_df.loc[:, ~pair_df.columns.astype(str).str.startswith("Unnamed:")].copy()

    required_columns = {"weather", "time_of_day", "count"}
    missing_columns = required_columns.difference(pair_df.columns)
    if missing_columns:
        raise ValueError(
            "Pair CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    pair_df = pair_df[["weather", "time_of_day", "count"]].copy()
    if pair_df[["weather", "time_of_day"]].isna().any().any():
        raise ValueError("Pair CSV contains missing weather or time_of_day values.")

    # Keep category spelling intact and strip only accidental surrounding whitespace.
    pair_df["weather"] = pair_df["weather"].astype(str).str.strip()
    pair_df["time_of_day"] = pair_df["time_of_day"].astype(str).str.strip()
    if (pair_df["weather"] == "").any() or (pair_df["time_of_day"] == "").any():
        raise ValueError("Pair CSV contains an empty weather or time_of_day value.")

    numeric_counts = pd.to_numeric(pair_df["count"], errors="raise")
    if ((numeric_counts % 1) != 0).any() or (numeric_counts < 0).any():
        raise ValueError("Pair CSV counts must be non-negative integers.")
    pair_df["count"] = numeric_counts.astype(int)

    duplicate_pairs = pair_df.duplicated(subset=["weather", "time_of_day"], keep=False)
    if duplicate_pairs.any():
        duplicates = pair_df.loc[duplicate_pairs, ["weather", "time_of_day"]]
        raise ValueError(
            "Pair CSV contains duplicate weather/time_of_day rows: "
            f"{duplicates.to_dict(orient='records')}"
        )

    return pair_df


def load_intersection_metadata(intersection_csv_path: str) -> pd.DataFrame:
    """Load and validate the 10K/100K image-metadata intersection CSV.

    The CSV is expected to use the schema emitted by
    ``find_100k_10k_intersection.py``. Only fields needed for pair-wise
    evaluation are retained after validation.

    Steps
    -----
    1. Read the intersection CSV and require the image ID, environment labels,
       label/image split metadata, same-split flag, and image path.
    2. Normalize string fields and reject missing or duplicate image IDs.
    3. Verify every ``image_path`` basename matches its declared ``id``.
    4. Rename ``timeofday`` to ``time_of_day`` for alignment with the pair CSV.

    Parameters
    ----------
    intersection_csv_path : str
        Path to ``bdd100k_10k_images_with_100k_metadata_intersection.csv``.

    Returns
    -------
    pd.DataFrame
        Validated table with ``id``, ``weather``, ``time_of_day``, split
        metadata, and ``metadata_image_path`` columns.

    Raises
    ------
    FileNotFoundError
        If ``intersection_csv_path`` does not exist.
    ValueError
        If required columns are missing, values are empty, IDs are duplicated,
        or an image path does not match its declared ID.
    """
    # This CSV is the image-level source of truth for environment metadata, so
    # validate it before reconstructing any evaluation subset.
    if not os.path.isfile(intersection_csv_path):
        raise FileNotFoundError(
            f"Intersection CSV not found: {intersection_csv_path}"
        )

    intersection_df = pd.read_csv(intersection_csv_path, dtype=str)
    required_columns = {
        "id",
        "weather",
        "timeofday",
        "label_split",
        "image_split",
        "same_split",
        "image_path",
    }
    missing_columns = required_columns.difference(intersection_df.columns)
    if missing_columns:
        raise ValueError(
            "Intersection CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    intersection_df = intersection_df[
        [
            "id",
            "weather",
            "timeofday",
            "label_split",
            "image_split",
            "same_split",
            "image_path",
        ]
    ].copy()
    if intersection_df.isna().any().any():
        raise ValueError(
            "Intersection CSV contains missing values in required columns."
        )

    # Preserve category spelling exactly while removing accidental surrounding
    # whitespace introduced by manual CSV editing.
    for column in [
        "id",
        "weather",
        "timeofday",
        "label_split",
        "image_split",
        "same_split",
        "image_path",
    ]:
        intersection_df[column] = intersection_df[column].str.strip()
        if (intersection_df[column] == "").any():
            raise ValueError(
                f"Intersection CSV contains an empty value in column '{column}'."
            )

    duplicate_ids = intersection_df["id"].duplicated(keep=False)
    if duplicate_ids.any():
        duplicate_examples = (
            intersection_df.loc[duplicate_ids, "id"].drop_duplicates().head(10).tolist()
        )
        raise ValueError(
            "Intersection CSV contains duplicate image IDs, e.g. "
            f"{duplicate_examples}"
        )

    # The producer writes the same BDD image ID into both ``id`` and the JPG
    # basename. Checking that invariant catches stale or manually mixed rows.
    path_ids = intersection_df["image_path"].map(
        lambda image_path: os.path.splitext(os.path.basename(image_path))[0]
    )
    mismatch_mask = path_ids != intersection_df["id"]
    if mismatch_mask.any():
        mismatch_rows = intersection_df.loc[
            mismatch_mask, ["id", "image_path"]
        ].head(10)
        raise ValueError(
            "Intersection CSV contains image_path/id mismatches, e.g. "
            f"{mismatch_rows.to_dict(orient='records')}"
        )

    intersection_df = intersection_df.rename(
        columns={
            "timeofday": "time_of_day",
            "image_path": "metadata_image_path",
        }
    )
    return intersection_df


def attach_intersection_metadata(
    test_df: pd.DataFrame,
    intersection_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach intersection weather/time-of-day metadata to held-out test rows.

    The original evaluators derive their test set from the BDD100k 10K training
    images. The intersection CSV covers only the subset of 10K images that also
    has matching 100K metadata, so an inner join is intentional here.

    Steps
    -----
    1. Derive the BDD image ID from each held-out JPEG filename.
    2. Inner-join the held-out test set with the validated intersection metadata.
    3. Verify every matched row is a 10K training image and that both image
       basenames agree.
    4. Return only the matched rows needed for pair-wise evaluation.

    Parameters
    ----------
    test_df : pd.DataFrame
        Held-out segmentation rows containing ``image_paths`` and ``mask_paths``.
    intersection_df : pd.DataFrame
        Validated 10K/100K intersection metadata keyed by ``id``.

    Returns
    -------
    pd.DataFrame
        Matched held-out rows augmented with ``id``, ``weather``, and
        ``time_of_day``.

    Raises
    ------
    ValueError
        If no held-out images are present in the intersection, a matched row is
        not from the 10K training split, or image basenames disagree.
    """
    held_out_df = test_df.copy()
    held_out_df["id"] = held_out_df["image_paths"].map(
        lambda image_path: os.path.splitext(os.path.basename(image_path))[0]
    )

    # Unmatched held-out images are intentionally excluded: by construction the
    # pair CSV describes only images in the 10K/100K metadata intersection.
    matched_df = held_out_df.merge(
        intersection_df,
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if matched_df.empty:
        raise ValueError(
            "No held-out segmentation images were found in the intersection CSV."
        )

    wrong_split_mask = matched_df["image_split"] != "train"
    if wrong_split_mask.any():
        wrong_rows = matched_df.loc[
            wrong_split_mask, ["id", "image_split"]
        ].head(10)
        raise ValueError(
            "Held-out evaluation rows must come from the 10K training split, "
            f"but the intersection CSV disagrees for: {wrong_rows.to_dict(orient='records')}"
        )

    test_basenames = matched_df["image_paths"].map(os.path.basename)
    metadata_basenames = matched_df["metadata_image_path"].map(os.path.basename)
    basename_mismatch = test_basenames != metadata_basenames
    if basename_mismatch.any():
        mismatch_rows = matched_df.loc[
            basename_mismatch, ["id", "image_paths", "metadata_image_path"]
        ].head(10)
        raise ValueError(
            "Held-out image paths do not match the intersection image paths, e.g. "
            f"{mismatch_rows.to_dict(orient='records')}"
        )

    return matched_df[
        ["image_paths", "mask_paths", "id", "weather", "time_of_day"]
    ].reset_index(drop=True)


def build_pair_test_frames(
    annotated_test_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> List[Tuple[str, str, int, int, pd.DataFrame]]:
    """Build and validate one held-out DataFrame per evaluation condition pair.

    Steps
    -----
    1. Remove rows whose weather or time of day is ``undefined`` because the
       analyzer excludes those rows before writing the pair CSV.
    2. Count every observed pair in the held-out evaluation population.
    3. Require the pair CSV to contain exactly the same pairs with exactly the
       same counts.
    4. Build one segmentation DataFrame per validated pair.

    Parameters
    ----------
    annotated_test_df : pd.DataFrame
        Held-out rows matched to the 10K/100K intersection metadata.
    pair_df : pd.DataFrame
        Validated pair specification generated from this same held-out
        evaluation population.

    Returns
    -------
    List[Tuple[str, str, int, int, pd.DataFrame]]
        Tuples of weather, time of day, pair-CSV count, evaluation count, and
        matching segmentation sample rows. The two counts are guaranteed equal.

    Raises
    ------
    ValueError
        If no defined evaluation rows exist, pair sets differ, or any pair CSV
        count differs from the number of held-out samples actually evaluated.
    """
    # The analyzer excludes undefined environment labels before counting pairs,
    # so validation must use the identical population here.
    evaluable_df = annotated_test_df.loc[
        (annotated_test_df["weather"] != "undefined")
        & (annotated_test_df["time_of_day"] != "undefined")
    ].copy()
    if evaluable_df.empty:
        raise ValueError(
            "No held-out metadata rows have defined weather and time_of_day values."
        )

    # Build maps for both inputs so missing, extra, and count-mismatched pairs
    # all fail before any checkpoint or GPU work begins.
    actual_counts = (
        evaluable_df.groupby(["weather", "time_of_day"], sort=False)
        .size()
        .to_dict()
    )
    expected_counts = {
        (str(row["weather"]), str(row["time_of_day"])): int(row["count"])
        for _, row in pair_df.iterrows()
    }

    actual_keys = set(actual_counts)
    expected_keys = set(expected_counts)
    missing_pairs = sorted(actual_keys - expected_keys)
    extra_pairs = sorted(expected_keys - actual_keys)
    mismatches: List[str] = []
    for pair_key in sorted(actual_keys & expected_keys):
        actual_count = int(actual_counts[pair_key])
        expected_count = expected_counts[pair_key]
        if actual_count != expected_count:
            mismatches.append(
                f"{pair_key[0]}/{pair_key[1]}: pair CSV={expected_count}, "
                f"evaluation={actual_count}"
            )

    if missing_pairs or extra_pairs or mismatches:
        details: List[str] = []
        if missing_pairs:
            details.append(
                "missing from pair CSV: "
                + ", ".join(f"{weather}/{time_of_day}" for weather, time_of_day in missing_pairs)
            )
        if extra_pairs:
            details.append(
                "not present in evaluation subset: "
                + ", ".join(f"{weather}/{time_of_day}" for weather, time_of_day in extra_pairs)
            )
        if mismatches:
            details.append("count mismatches: " + "; ".join(mismatches))
        raise ValueError(
            "Pair CSV does not match the held-out evaluation population: "
            + " | ".join(details)
            + ". Regenerate ./data/weather_timeofday_pair_data.csv with "
            "analyze_100k_10k_intersection.py."
        )

    pair_frames: List[Tuple[str, str, int, int, pd.DataFrame]] = []

    for _, pair_row in pair_df.iterrows():
        weather = str(pair_row["weather"])
        time_of_day = str(pair_row["time_of_day"])
        pair_csv_count = int(pair_row["count"])

        # Exact matching avoids inventing aliases between environmental categories.
        pair_mask = (
            (evaluable_df["weather"] == weather)
            & (evaluable_df["time_of_day"] == time_of_day)
        )
        subset_df = evaluable_df.loc[
            pair_mask, ["image_paths", "mask_paths"]
        ].reset_index(drop=True)
        evaluation_count = len(subset_df)

        # Equality was validated above; keeping both values in the output makes
        # the data provenance explicit without allowing them to diverge silently.
        pair_frames.append(
            (
                weather,
                time_of_day,
                pair_csv_count,
                evaluation_count,
                subset_df,
            )
        )

    if not pair_frames:
        raise ValueError(
            "No weather/time_of_day pairs are available for held-out evaluation."
        )

    return pair_frames


def average_performance_metrics(
    run_metrics: List[Dict[str, float | List[float]]],
) -> Dict[str, float | List[float]]:
    """Average scalar and per-class performance metrics across repeated runs.

    Steps
    -----
    1. Iterate through the metric keys produced by the first run.
    2. Average scalar metrics directly and list metrics element-wise.
    3. Preserve the original metric dictionary structure.

    Parameters
    ----------
    run_metrics : List[Dict[str, float | List[float]]]
        Performance dictionaries produced by repeated evaluations.

    Returns
    -------
    Dict[str, float | List[float]]
        Averaged performance metric dictionary.

    Raises
    ------
    ValueError
        If no runs were supplied or a metric changes type between runs.
    """
    if not run_metrics:
        raise ValueError("At least one completed run is required for averaging.")

    averaged_metrics: Dict[str, float | List[float]] = {}
    for key in run_metrics[0]:
        values = [run[key] for run in run_metrics]
        if all(isinstance(value, float) for value in values):
            scalar_values = [cast(float, value) for value in values]
            averaged_metrics[key] = float(np.mean(scalar_values))
        elif all(isinstance(value, list) for value in values):
            averaged_metrics[key] = np.mean(np.asarray(values, dtype=float), axis=0).tolist()
        else:
            raise ValueError(f"Metric '{key}' changed type across evaluation runs.")

    return averaged_metrics


def flatten_performance_metrics(
    weather: str,
    time_of_day: str,
    source_count: int,
    evaluation_count: int,
    metrics: Dict[str, float | List[float]],
) -> Dict[str, str | int | float]:
    """Flatten one pair's performance metrics into a CSV-friendly row.

    Steps
    -----
    1. Store the pair labels, pair-CSV count, evaluation count, and scalar metrics.
    2. Expand per-class pixel accuracy and IoU lists into named columns.

    Parameters
    ----------
    weather : str
        Weather category for the evaluated subset.
    time_of_day : str
        Time-of-day category for the evaluated subset.
    source_count : int
        Count recorded in the pair CSV for this held-out evaluation pair.
    evaluation_count : int
        Number of held-out segmentation images actually evaluated for the pair;
        guaranteed to equal ``source_count``.
    metrics : Dict[str, float | List[float]]
        Averaged segmentation metrics for the pair.

    Returns
    -------
    Dict[str, str | int | float]
        Flat row ready for DataFrame/CSV export.
    """
    row: Dict[str, str | int | float] = {
        "weather": weather,
        "time_of_day": time_of_day,
        "pair_csv_count": source_count,
        "evaluation_count": evaluation_count,
        "pixel_accuracy": cast(float, metrics["pixel_accuracy"]),
        "iou_macro": cast(float, metrics["iou"]),
        "mean_iou": cast(float, metrics["mean_iou"]),
        "global_iou": cast(float, metrics["global_iou"]),
        "dice_macro": cast(float, metrics["dice"]),
    }

    class_accuracy = cast(List[float], metrics["class_pixel_accuracy"])
    class_iou = cast(List[float], metrics["class_iou"])
    for class_id in range(19):
        class_slug = CLASS_NAMES[class_id].replace(" ", "_")
        row[f"class_pixel_accuracy_{class_id}_{class_slug}"] = class_accuracy[class_id]
        row[f"class_iou_{class_id}_{class_slug}"] = class_iou[class_id]

    return row


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for pair-wise performance evaluation.

    Steps
    -----
    1. Require the model checkpoint path.
    2. Default the pair specification and 10K/100K intersection CSVs to their
       project ``./data`` locations while allowing explicit overrides.
    3. Add batch-size, repetition, output, and visualization controls.
    4. Parse and return the argument namespace.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate segmentation performance for each weather/time-of-day "
            "pair in the BDD100k 10K/100K metadata intersection."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PyTorch checkpoint file (.pt) to evaluate.",
    )
    parser.add_argument(
        "--pair-csv",
        type=str,
        default=Path.WEATHER_TIMEOFDAY_PAIR_CSV,
        help=(
            "Pair/count CSV. Defaults to "
            "./data/weather_timeofday_pair_data.csv."
        ),
    )
    parser.add_argument(
        "--intersection-csv",
        type=str,
        default=Path.METADATA_INTERSECTION_CSV,
        help=(
            "10K image / 100K metadata intersection CSV. Defaults to "
            "./data/bdd100k_10k_images_with_100k_metadata_intersection.csv."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Configuration.BATCH_SIZE,
        help="DataLoader batch size. Defaults to the configuration value.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of evaluation runs averaged per pair. Defaults to 5.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./model/performance_weather_timeofday.csv",
        help="Destination CSV for pair-wise metrics.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Export sample predictions from the union of positive-count pairs.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate segmentation performance for each requested condition pair.

    Steps
    -----
    1. Load the pair specification and 10K/100K intersection metadata.
    2. Reconstruct the held-out segmentation split used by the base evaluator
       and keep only rows present in the metadata intersection.
    3. Validate that every pair CSV count exactly matches the corresponding
       held-out evaluation count.
    4. Load the trained checkpoint only after the data mapping is proven valid.
    5. Evaluate each pair for the requested number of runs.
    6. Print and export the equal pair/evaluation counts and all metrics.
    7. Optionally visualize samples from the union of evaluated pairs.

    Raises
    ------
    ValueError
        If batch size or number of runs is not positive, or input mapping fails.
    """
    args = parse_arguments()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be greater than zero.")

    print("\n" + "=" * 76)
    print("PERFORMANCE BY WEATHER / TIME OF DAY")
    print("=" * 76)

    # Load both CSV inputs before model work; pair counts are validated only
    # after reconstructing the exact held-out segmentation population they describe.
    pair_df = load_pair_specification(args.pair_csv)
    intersection_df = load_intersection_metadata(args.intersection_csv)

    print("Loading BDD100k held-out test split...")
    _, _, test_df = load_dataset_from_files()
    annotated_test_df = attach_intersection_metadata(test_df, intersection_df)
    print(
        f"Matched {len(annotated_test_df)} of {len(test_df)} held-out images "
        "to the 10K/100K metadata intersection."
    )

    pair_frames = build_pair_test_frames(annotated_test_df, pair_df)
    print(
        "Validated pair CSV counts against the held-out evaluation population; "
        "pair_csv_count == evaluation_count for every pair."
    )
    print(
        f"Built {len(pair_frames)} evaluable pairs covering "
        f"{sum(pair[3] for pair in pair_frames)} held-out segmentation images."
    )

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, Configuration.DEVICE)
    print(f"Target execution device: {Configuration.DEVICE}")

    inference_transforms = A.Compose([
        A.Resize(height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH),
        ToTensorV2(),
    ])

    summary_rows: List[Dict[str, str | int | float]] = []
    evaluated_frames: List[pd.DataFrame] = []
    for weather, time_of_day, source_count, sample_count, pair_test_df in pair_frames:
        print("\n" + "-" * 76)
        print(
            f"Pair: weather={weather!r}, time_of_day={time_of_day!r}, "
            f"pair_csv_count={source_count}, evaluation_count={sample_count}"
        )

        # Isolating each pair in its own loader makes all confusion counts and
        # reductions specific to that environmental condition.
        pair_dataset = BDDSegmentationDataset(
            pair_test_df,
            transform=inference_transforms,
        )
        pair_loader = DataLoader(
            dataset=pair_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=Configuration.NUM_WORKERS,
        )

        run_metrics: List[Dict[str, float | List[float]]] = []
        for run_index in range(args.num_runs):
            run_metrics.append(
                evaluate_segmentation_metrics(
                    model,
                    pair_loader,
                    Configuration.DEVICE,
                )
            )
            print(f"  Run {run_index + 1}/{args.num_runs} complete.")

        averaged_metrics = average_performance_metrics(run_metrics)
        summary_rows.append(
            flatten_performance_metrics(
                weather,
                time_of_day,
                source_count,
                sample_count,
                averaged_metrics,
            )
        )
        evaluated_frames.append(pair_test_df)

        print(
            f"  Pixel Accuracy : "
            f"{cast(float, averaged_metrics['pixel_accuracy']):.4f}"
        )
        print(f"  IoU (macro)    : {cast(float, averaged_metrics['iou']):.4f}")
        print(
            f"  Global IoU     : "
            f"{cast(float, averaged_metrics['global_iou']):.4f}"
        )
        print(f"  Mean IoU       : {cast(float, averaged_metrics['mean_iou']):.4f}")
        print(f"  Dice (macro)   : {cast(float, averaged_metrics['dice']):.4f}")

    # Export all scalar and class-wise values so downstream comparison is lossless.
    output_directory = os.path.dirname(args.output_csv)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output_csv, index=False)
    print("\n" + "=" * 76)
    print(f"Saved pair-wise performance metrics to: {args.output_csv}")

    if args.visualize:
        combined_test_df = pd.concat(evaluated_frames, ignore_index=True)
        output_path = "./model/predictions_weather_timeofday.png"
        print(f"Exporting sample predictions to {output_path}...")
        visualize_predictions(
            model=model,
            test_df=combined_test_df,
            device=Configuration.DEVICE,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
