"""Performance evaluation script for trained semantic segmentation models.

This module loads a trained PyTorch segmentation model checkpoint (a full
``nn.Module`` serialized by the training scripts) and evaluates it on the BDD100k
test split. It reuses the exact segmentation metrics logic from the training
scripts (``evaluate_segmentation_metrics`` from ``train_teacher_normal.py``) so
that the numbers reported here are directly comparable to the ones logged at the
end of training.
"""

import argparse
import glob
from typing import Dict, List, Tuple, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, UInt8, jaxtyped
import numpy as np
import pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import torch
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


def find_train_mask_path_from_image(complete_image_path: str) -> str:
    """Locate the training mask corresponding to a training image.

    Steps
    -----
    1. Delegate to :func:`find_mask_path_from_image` using
       ``Path.SEGMENTATION_MASK_TRAIN_PATH``.

    Parameters
    ----------
    complete_image_path : str
        Full path to a training image JPEG.

    Returns
    -------
    str
        Full path to the corresponding training mask PNG.
    """
    return find_mask_path_from_image(complete_image_path, Path.SEGMENTATION_MASK_TRAIN_PATH)


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
    # Load train masks, dropping any that do not match the expected resolution.
    train_mask_paths = glob.glob(f"{Path.SEGMENTATION_MASK_TRAIN_PATH}/*.png")
    problematic_masks: List[str] = []
    for complete_mask_path in train_mask_paths:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_masks.append(complete_mask_path)
                train_mask_paths.remove(complete_mask_path)
    print(f"Problematic masks: {problematic_masks}")

    train_image_paths = list(map(find_train_image_path_from_mask, train_mask_paths))
    problematic_images: List[str] = []
    for complete_image_path in train_image_paths:
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_images.append(complete_image_path)
                train_image_paths.remove(complete_image_path)
                train_mask_paths.remove(find_train_mask_path_from_image(complete_image_path))
    print(f"Problematic images: {problematic_images}")

    train_test_df = pd.DataFrame({
        "image_paths": train_image_paths,
        "mask_paths": train_mask_paths,
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
    pixel_accuracy = smp.metrics.accuracy(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()

    # Class-wise pixel accuracy: one accuracy value per class (no reduction).
    # ``reduction=None`` returns a ``(num_classes,)`` tensor whose element ``i``
    # is the accuracy of class ``i``.
    class_pixel_accuracy = smp.metrics.accuracy(tp_total, fp_total, fn_total, tn_total).tolist()

    # IoU (Jaccard index) and Dice (F1) macro-averaged over classes.
    iou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()
    dice = smp.metrics.f1_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()

    # Class-wise IoU: one IoU value per class (no reduction). ``reduction=None``
    # returns a ``(num_classes,)`` tensor whose element ``i`` is the IoU of
    # class ``i``.
    class_iou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total).tolist()

    # Mean IoU: average of the class-wise IoU values (macro-averaging).
    mean_iou = np.mean(class_iou)

    # Dataset-level global (micro) IoU
    global_iou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "class_pixel_accuracy": list(class_pixel_accuracy),
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the performance evaluation script.

    Steps
    -----
    1. Configure an ArgumentParser with an optional checkpoint path argument.
    2. Add an optional batch-size argument to override the configuration.
    3. Add an optional number-of-runs argument controlling how many times
       evaluation is repeated before averaging.
    4. Parse and return the argument namespace.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation metrics of a trained model on the BDD100k test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PyTorch checkpoint file (.pt) to evaluate.",
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
        help="Number of evaluation runs over which metrics are averaged. Defaults to 5.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and report full segmentation metrics on the test split.

    Steps
    -----
    1. Parse the checkpoint, batch-size, and number-of-runs arguments.
    2. Load the trained model onto the target device.
    3. Load the BDD100k test partition with the training-compatible transforms.
    4. Build the evaluation DataLoader.
    5. Run :func:`evaluate_segmentation_metrics` ``N`` times and average every
       reported metric (scalars and per-class lists alike) so the logged numbers
       are robust to single-run variance.
    """
    args = parse_arguments()

    print("\n" + "=" * 68)
    print("PERFORMANCE EVALUATION PIPELINE")
    print("=" * 68)

    # Load the checkpoint with the whole nn.Module serialized by the trainers.
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, Configuration.DEVICE)
    print(f"Target execution device: {Configuration.DEVICE}")

    # Recreate the same train/val/test partitioning used during training so the
    # test split evaluated here is identical to the one held out at training time.
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
        shuffle=False,  # deterministic ordering for reproducible evaluation
        num_workers=Configuration.NUM_WORKERS,
    )

    # Compute the same metrics the training scripts report on their best
    # checkpoint so the numbers can be compared one-to-one. Evaluate multiple
    # times and average so the reported metrics are stable and robust to any
    # single-run variance.
    num_runs = args.num_runs
    print(f"Evaluating over {num_runs} run(s)...")
    run_metrics: List[Dict[str, float | List[float]]] = []
    for run_idx in range(num_runs):
        metrics = evaluate_segmentation_metrics(
            model, test_loader, Configuration.DEVICE
        )
        run_metrics.append(metrics)
        print(f"  Run {run_idx + 1}/{num_runs} complete.")

    # Reduce the per-run results into a single averaged metric dictionary. Scalar
    # metrics (e.g. pixel accuracy, IoU, Dice) are averaged directly, while the
    # per-class lists are averaged element-wise so class ``i`` still reports the
    # mean of class ``i`` across runs.
    averaged_metrics: Dict[str, float | List[float]] = {}
    for key in run_metrics[0]:
        values = [run[key] for run in run_metrics]
        if all(isinstance(value, float) for value in values):
            averaged_metrics[key] = float(np.mean(values))
        else:
            averaged_metrics[key] = list(np.mean(np.array(values), axis=0))

    print('\nFinal test-set metrics (averaged over %d runs):' % num_runs)
    print(f'  Pixel Accuracy : {averaged_metrics["pixel_accuracy"]:.4f}')
    print(f'  IoU (macro)    : {averaged_metrics["iou"]:.4f}')
    print(f'  Global IoU     : {averaged_metrics["global_iou"]:.4f}')
    print(f'  Mean IoU       : {averaged_metrics["mean_iou"]:.4f}')
    print(f'  Dice (macro)   : {averaged_metrics["dice"]:.4f}')
    print('  Class-wise Pixel Accuracy:')
    for class_id, acc in enumerate(cast(List[float], averaged_metrics["class_pixel_accuracy"])):
        print(f'    {class_id:>2} {CLASS_NAMES[class_id]:<14} : {acc:.4f}')
    print('  Class-wise IoU:')
    for class_id, class_iou in enumerate(cast(List[float], averaged_metrics["class_iou"])):
        print(f'    {class_id:>2} {CLASS_NAMES[class_id]:<14} : {class_iou:.4f}')


if __name__ == "__main__":
    main()
