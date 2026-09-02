import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
import torch.optim.lr_scheduler as lr_scheduler

from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
from typing import Dict, List, cast

from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

from jaxtyping import Float, UInt8, jaxtyped
from beartype import beartype


# Shape aliases that document the tensor layout at each stage of the pipeline.
#
# ``h``/``w`` are the spatial dimensions, ``b`` is the batch size and ``c`` is
# the number of channels/classes. These aliases are enforced at runtime by
# ``jaxtyped`` + ``beartype``, so a shape mismatch raises a ``TypeCheckError``
# instead of a confusing downstream broadcast error.
ImageTensor = Float[Tensor, "3 h w"]  # single image, channel-first layout
MaskTensor = Float[Tensor, "c h w"]  # one-hot mask, ``c == NUM_CLASSES``
BatchImage = Float[Tensor, "b 3 h w"]  # collated batch of images
BatchMask = Float[Tensor, "b c h w"]  # collated batch of one-hot masks
Logits = Float[Tensor, "b c h w"]  # model output, ``c == NUM_CLASSES``
ClassMask = Float[Tensor, "c h w"]  # per-sample probability/binary mask, ``c`` channels
Scalar = Float[Tensor, ""]  # scalar (0-dim) tensor
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


class Configuration:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_DEVICES = 1
    NUM_WORKERS= 2

    NUM_CLASSES = 20
    EPOCHS = 20
    BATCH_SIZE = (
        4 if torch.cuda.device_count() < 2
        else (4 * torch.cuda.device_count())
    )
    LR = 1e-4
    PATIENCE = 8

    APPLY_SHUFFLE=True
    SEED = 768
    # ``IMAGE_HEIGHT``/``IMAGE_WIDTH`` describe the spatial extent of a sample.
    # The BDD100k images used here are 720 rows (height) by 1280 columns
    # (width); note the previous names were swapped, which made shape code
    # downstream ambiguous even though the numbers happened to line up.
    # However, we're going to downscale the images.
    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 640
    CHANNELS = 3 # RGB


class ImagePath:
    BASE = "./data/bdd100k"
    SEGMENTATION_MASK_LABEL_FOLDER = BASE + "/segmentation_maps/color_labels"
    SEGMENTATION_MASK_TRAIN_PATH = SEGMENTATION_MASK_LABEL_FOLDER + "/train"
    SEGMENTATION_MASK_VAL_PATH = SEGMENTATION_MASK_LABEL_FOLDER + "/val"

    IMAGE_TRAIN_PATH = BASE + "/images_10k/train"
    IMAGE_VAL_PATH = BASE + "/images_10k/val"


class BDDSegmentationDataset(Dataset[tuple[ImageTensor, MaskTensor]]):
    def __init__(self, df: pd.DataFrame, transform: A.Compose | None = None):
        super(BDDSegmentationDataset, self).__init__()

        self.image_paths: List[str] = df["image_paths"].to_list()
        self.mask_paths: List[str] = df["mask_paths"].to_list()
        self.transform = transform


    @jaxtyped(typechecker=beartype)
    def load_sample(self, index: int) -> tuple[NumpyImage, ClassIndexArray]:
        """Load the image and its per-pixel class-index mask at ``index``.

        The image is opened as RGB and normalised to ``[0, 1]``. The mask PNG
        is likewise forced to RGB (some BDD100k color-labels carry an alpha
        channel) before being collapsed from the RGB color-label format to a
        single class id per pixel via :func:`color_label_to_class_index`. This
        class-index representation is what the one-hot encoder and Dice loss
        expect downstream.

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
        tuple[NumpyImage, ClassIndexArray]
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
        return len(self.image_paths)


    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> tuple[ImageTensor, MaskTensor]:
        """Return the transformed ``(image, mask)`` pair at position ``index``.

        ``ToTensorV2`` converts the image from ``(H, W, 3)`` to ``(3, H, W)``
        and leaves the 2-D class-index mask as ``(H, W)``. The mask is then
        one-hot encoded to ``(NUM_CLASSES, H, W)`` so that its channel axis
        lines up with the model logits and the Dice loss. The DataLoader adds
        the batch dimension when collating samples.

        Parameters
        ----------
        index : int
            Zero-based position of the sample to load.

        Returns
        -------
        tuple[ImageTensor, MaskTensor]
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
        # so the mask matches the model's (B, NUM_CLASSES, H, W) output and the
        # Dice loss. ``one_hot`` needs int64 input, hence the cast.
        mask_one_hot = torch.nn.functional.one_hot(
            transformed["mask"].to(torch.int64), Configuration.NUM_CLASSES
        ).permute(2, 0, 1).float()

        return transformed["image"], mask_one_hot


def find_image_path_from_mask(complete_mask_path: str, base_image_path: str) -> str:
    file_path_split = complete_mask_path.split("/")
    mask_file_name = file_path_split[-1].split("_")[0]

    image_path = base_image_path + "/" + mask_file_name + ".jpg"
    return image_path


def find_train_image_path_from_mask(complete_mask_path: str) -> str:
    return find_image_path_from_mask(complete_mask_path, ImagePath.IMAGE_TRAIN_PATH)


def find_val_image_path_from_mask(complete_mask_path: str) -> str:
    return find_image_path_from_mask(complete_mask_path, ImagePath.IMAGE_VAL_PATH)


def load_dataset_from_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # load train, then split into train-test
    train_mask_paths = glob.glob(f"{ImagePath.SEGMENTATION_MASK_TRAIN_PATH}/*.png")
    train_image_paths = list(map(find_train_image_path_from_mask, train_mask_paths))

    train_test_df = pd.DataFrame({
        "image_paths": train_image_paths,
        "mask_paths": train_mask_paths,
    })
    train_df, test_df = train_test_split(train_test_df, test_size=0.2, random_state=Configuration.SEED)

    # load val
    val_mask_paths = glob.glob(f"{ImagePath.SEGMENTATION_MASK_VAL_PATH}/*.png")
    val_image_paths = list(map(find_val_image_path_from_mask, val_mask_paths))
    val_df = pd.DataFrame({
        "image_paths": val_image_paths,
        "mask_paths": val_mask_paths,
    })

    return train_df, val_df, test_df


@jaxtyped(typechecker=beartype)
def forward(model: nn.Module, x: BatchImage) -> Logits:
    """Run a single forward pass and assert the input/output shapes.

    Centralising the forward pass here lets ``jaxtyping`` verify that the
    input batch is always ``(B, 3, H, W)`` and that the model produces
    ``(B, NUM_CLASSES, H, W)`` logits, which is where shape confusion most
    often arises.

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


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss for semantic segmentation.

    ``nn.CrossEntropyLoss`` expects integer class-id targets of shape ``(B, H,
    W)``, which is incompatible with a segmentation pipeline that wishes to use
    a Dice objective over per-class probabilities. This module instead takes
    raw ``(B, C, H, W)`` logits and one-hot ``(B, C, H, W)`` targets, applies a
    softmax over the class axis and returns ``1 - mean(Dice)`` averaged over
    classes and the batch.

    Parameters
    ----------
    smooth : float, optional
        Additive smoothing applied to the Dice numerator and denominator to
        avoid division by zero. Defaults to 1.0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Logits, target: BatchMask) -> Scalar:
        """Compute the loss value.

        Parameters
        ----------
        logits : Logits
            Raw model output of shape ``(B, C, H, W)``.
        target : BatchMask
            One-hot ground-truth mask of shape ``(B, C, H, W)``.

        Returns
        -------
        Scalar
            The scalar loss value (1 - mean Dice), differentiable w.r.t.
            ``logits``.
        """
        # Softmax over the class axis yields per-pixel probabilities that sum
        # to 1 across classes, matching the one-hot target distribution.
        probs = torch.softmax(logits, dim=1)

        # Reduce the spatial axes to get a Dice coefficient per (sample, class)
        # before averaging: this treats every class equally regardless of the
        # number of pixels it occupies, which prevents the road/sky classes
        # from dominating the loss.
        intersection = (probs * target).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


@jaxtyped(typechecker=beartype)
def execute_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple[float, float]:

    # Set model into training mode
    model.train()

    # Initialize train loss & accuracy
    train_loss, train_dice = 0.0, 0.0

    # Execute training loop over train dataloader
    for _, (X, y) in enumerate(dataloader):
        # Load data onto target device
        X, y = X.to(device), y.to(device)

        # Feed-forward and compute metrics
        y_pred = forward(model, X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Reset Gradients & Backpropagate Loss
        optimizer.zero_grad()
        loss.backward()

        # Update Model Gradients
        optimizer.step()

        # Compute Batch Metrics
        # ``y`` is one-hot and ``y_pred`` holds raw logits, so run a softmax
        # over the class axis to obtain per-class probabilities in the same
        # (B, NUM_CLASSES, H, W) space before computing soft Dice.
        predicted = torch.softmax(y_pred, dim=1)

        eps = 1e-8
        train_dice += (
            (2 * (y * predicted).sum() + eps) /
            ((y + predicted).sum() + eps)
        ).cpu().item()


    # Compute Step Metrics
    train_loss = train_loss / len(dataloader)
    train_dice = train_dice / len(dataloader)

    return train_loss, train_dice


@jaxtyped(typechecker=beartype)
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    loss_fn: nn.Module,
    device: torch.device
) -> tuple[float, float]:

    # Set model into eval mode
    model.eval()

    # Initialize eval loss & accuracy
    eval_loss, eval_dice = 0.0, 0.0

    # Active inferene context manager
    with torch.inference_mode():
        # Execute eval loop over dataloader
        for _, (X, y) in enumerate(dataloader):
            # Load data onto target device
            X, y = X.to(device), y.to(device)

            # Feed-forward and compute metrics
            y_pred = forward(model, X)
            loss = loss_fn(y_pred, y)
            eval_loss += loss.item()

            # Compute Batch Metrics
            # Softmax gives per-class probabilities over the class axis, matching
            # the one-hot ``y`` shape so soft Dice is well-defined.
            predicted = torch.softmax(y_pred, dim=1)

            eps = 1e-8
            eval_dice += (
                (2 * (y * predicted).sum() + eps) /
                ((y + predicted).sum() + eps)
            ).cpu().item()

    # Compute Step Metrics
    eval_loss = eval_loss / len(dataloader)
    eval_dice = eval_dice / len(dataloader)

    return eval_loss, eval_dice


@jaxtyped(typechecker=beartype)
def train(
    model: nn.Module,
    train_dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    eval_dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler.ReduceLROnPlateau | None,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device
) -> Dict[str, List[float]]:

    # Initialize training session
    session: Dict[str, List[float]] = {
        'loss'            : [],
        'dice_score'      : [],
        'eval_loss'       : [],
        'eval_dice_score' : []
    }

    # Training loop
    for epoch in tqdm(range(epochs)):
        # Execute Epoch
        print(f'\nEpoch {epoch + 1}/{epochs}')
        train_loss, train_dice = execute_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            device
        )

        # Evaluate Model
        eval_loss, eval_dice = evaluate(
            model,
            eval_dataloader,
            loss_fn,
            device
        )

        # Execute schedular step
        current_lr = 0
        if scheduler:
            scheduler.step(eval_loss)
            current_lr = optimizer.param_groups[0]['lr']

        # Log Epoch Metrics
        log_text = f'loss: {train_loss:.4f} - dice_score: {train_dice:.4f} - eval_loss: {eval_loss:.4f} - eval_dice_score: {eval_dice:.4f}'

        if scheduler:
            print(log_text + f' - lr: {current_lr}')
        else:
            print(log_text)

        # Record Epoch Metrics
        session['loss'].append(train_loss)
        session['dice_score'].append(train_dice)
        session['eval_loss'].append(eval_loss)
        session['eval_dice_score'].append(eval_dice)

    # Return Session Metrics
    return session


def plot_training_curves(
    history: Dict[str, List[float]],
    fig_size: tuple[int, int] = (20, 10)
) -> None:

    loss = np.array(history['loss'])
    val_loss = np.array(history['eval_loss'])

    dice_coeff = np.array(history['dice_score'])
    val_dice_coeff = np.array(history['eval_dice_score'])

    epochs = range(len(history['loss']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)

    # Plot loss
    ax1.plot(epochs, loss, label='training_loss', marker='o', color='C5')
    ax1.plot(epochs, val_loss, label='eval_loss', marker='o', color='C6')

    # Fill area between losses
    ax1.fill_between(epochs, loss, val_loss, where=(loss > val_loss), color='C5', alpha=0.4, interpolate=True)
    ax1.fill_between(epochs, loss, val_loss, where=(loss < val_loss), color='C6', alpha=0.4, interpolate=True)

    # Add Text & Formats
    ax1.set_title('Loss (Lower Means Better)', fontsize=22)
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Loss', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=14)

    # Plot metric
    ax2.plot(epochs, dice_coeff, label='training_dice_score', marker='o', color='C5')
    ax2.plot(epochs, val_dice_coeff, label='eval_dice_score', marker='o', color='C6')

    # Fill area between metrics
    ax2.fill_between(epochs, dice_coeff, val_dice_coeff, where=(dice_coeff > val_dice_coeff), color='C5', alpha=0.4, interpolate=True)
    ax2.fill_between(epochs, dice_coeff, val_dice_coeff, where=(dice_coeff < val_dice_coeff), color='C6', alpha=0.4, interpolate=True)

    # Add Text & Formats
    ax2.set_title('Dice Score (Higher Means Better)', fontsize=22)
    ax2.set_xlabel('Epochs', fontsize=18)
    ax2.set_ylabel('Dice Score', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(fontsize=14)
    sns.despine()


@jaxtyped(typechecker=beartype)
def precision_(y_true: ClassMask, y_pred: ClassMask) -> Scalar:
    """Compute mean precision (intersection over predicted positives).

    Both operands must share the same ``(C, H, W)`` shape; the axis ``C`` is
    reduced by the summation and the result is a scalar.

    Parameters
    ----------
    y_true : ClassMask
        Ground-truth mask with shape ``(C, H, W)``.
    y_pred : ClassMask
        Predicted mask with shape ``(C, H, W)``.

    Returns
    -------
    Scalar
        Mean precision across the class dimension.
    """
    intersection = (y_true * y_pred).sum()
    total_predicted_pixels = y_pred.sum()
    return (intersection / total_predicted_pixels).mean()


@jaxtyped(typechecker=beartype)
def recall_(y_true: ClassMask, y_pred: ClassMask) -> Scalar:
    """Compute mean recall (intersection over true positives).

    Both operands must share the same ``(C, H, W)`` shape; the axis ``C`` is
    reduced by the summation and the result is a scalar.

    Parameters
    ----------
    y_true : ClassMask
        Ground-truth mask with shape ``(C, H, W)``.
    y_pred : ClassMask
        Predicted mask with shape ``(C, H, W)``.

    Returns
    -------
    Scalar
        Mean recall across the class dimension.
    """
    intersection = (y_true * y_pred).sum()
    total_true_pixels = y_true.sum()
    return (intersection / total_true_pixels).mean()


@jaxtyped(typechecker=beartype)
def dice_score(y_true: ClassMask, y_pred: ClassMask) -> Scalar:
    """Compute the Sorensen-Dice coefficient for a single mask pair.

    Both operands must share the same ``(C, H, W)`` shape, which ``jaxtyped``
    enforces at runtime before the element-wise product is evaluated.

    Parameters
    ----------
    y_true : ClassMask
        Ground-truth mask with shape ``(C, H, W)``.
    y_pred : ClassMask
        Predicted mask with shape ``(C, H, W)``.

    Returns
    -------
    Scalar
        Scalar Dice coefficient, smoothed by ``eps`` to avoid division by zero.
    """
    eps = 1e-8
    intersection = (y_true * y_pred).sum()
    summation = (y_true + y_pred).sum()

    return ((2 * intersection) / (summation + eps))


@jaxtyped(typechecker=beartype)
def jaccard_index(y_true: ClassMask, y_pred: ClassMask) -> Scalar:
    """Compute the Jaccard index (IoU) for a single mask pair.

    Both operands must share the same ``(C, H, W)`` shape, which ``jaxtyped``
    enforces at runtime before the element-wise product is evaluated.

    Parameters
    ----------
    y_true : ClassMask
        Ground-truth mask with shape ``(C, H, W)``.
    y_pred : ClassMask
        Predicted mask with shape ``(C, H, W)``.

    Returns
    -------
    Scalar
        Scalar IoU, smoothed by ``eps`` to avoid division by zero.
    """
    eps = 1e-8
    intersection = (y_true * y_pred).sum()
    union = (y_true + y_pred).sum() - intersection

    return (intersection / (union + eps))


def compute_metrics(
    model: nn.Module,
    sample_loader: DataLoader[tuple[ImageTensor, MaskTensor]],
    device: torch.device
) -> Dict[str, List[float]]:

    # Initiate Metrics Dict
    metrics: Dict[str, List[float]] = {
        'IoU'           : [],
        'dice_score'    : [],
    }

    # Set model into eval mode
    model.eval()

    # Active inferene context manager
    with torch.inference_mode():
        # Execute eval loop over dataloader
        for _, (X, y) in enumerate(tqdm(sample_loader)):
            # Load data onto target device
            X, y = X.to(device), y.to(device)

            # Feed-forward Input
            y_pred = forward(model, X)

            # Generate Predicted Masks
            # Softmax yields per-class probabilities matching the one-hot ``y``;
            # these are passed straight to the soft Dice/IoU helpers.
            predicted = torch.softmax(y_pred, dim=1)

            # Compute Batch Metrics For Each Mask
            for true_mask, pred_mask in zip(y, predicted, strict=True):
                iou = jaccard_index(true_mask, pred_mask).cpu().item()
                dice = dice_score(true_mask, pred_mask).cpu().item()

                # Record metrics
                metrics['dice_score'].append(dice)
                metrics['IoU'].append(iou)

    return metrics


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
    model:nn.Module,
    test_df:pd.DataFrame,
    device:torch.device,
    num_samples:int = 4,
    output_path:str = "./predictions.png",
) -> None:

    """Render random test samples next to their true and predicted masks.

    A few rows of ``test_df`` are sampled, every image is run through
    ``model``, and a grid with three columns (image / image + true mask /
    image + predicted mask) is exported to a PNG file. The model's ``(20, H,
    W)`` logits are reduced to a single class id per pixel via ``argmax`` so
    they can be coloured with ``CLASS_COLORS`` and compared to the
    ground-truth color labels.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model returning ``(B, 20, H, W)`` logits.
    test_df : pd.DataFrame
        DataFrame carrying the ``image_paths`` and ``mask_paths`` columns.
    device : torch.device
        Device used to run inference.
    num_samples : int, optional
        Number of random samples to visualise. Defaults to 4.
    output_path : str, optional
        Destination of the exported PNG. Defaults to ``"./predictions.png"``.

    Raises
    ------
    ValueError
        If ``test_df`` has no rows to sample.
    """
    # Sample a fixed number of random rows (or fewer if the frame is small)
    # so the visualisation changes with every call while staying reproducible
    # thanks to the fixed random state.
    sample_df = test_df.sample(
        n=min(num_samples, len(test_df)),
        random_state=Configuration.SEED,
    ).reset_index(drop=True)

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
    model.eval()

    for row in range(num_rows):
        # Load the raw pair: the image as an (H, W, 3) float array in [0, 1]
        # and the mask as an (H, W) class-index array.
        image, class_mask = sample_ds.load_sample(row)

        # Replicate the ToTensorV2 conversion: transpose HWC -> CHW, add a
        # batch dim and move to the device so the model sees the same format
        # it received during training.
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).contiguous()
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Foward pass, then collapse the 20-class logits to one class id per
        # pixel so the output can be colourised.
        with torch.inference_mode():
            logits = model(image_tensor)
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

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    # Print current Torch package versions
    print('Package versions:')
    print('*'*26)
    print(f'torch \t\t - {torch.__version__}')
    print(f'torchvision \t - {torchvision.__version__}')

    train_df, val_df, test_df = load_dataset_from_files()

    train_transforms = A.Compose([
        A.Resize(height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH),
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        # The mask is now a 2-D class-index map, so ``ToTensorV2`` needs no
        # ``transpose_mask``: it leaves the (H, W) mask as-is and only converts
        # the image to (C, H, W).
        ToTensorV2(),
    ])

    inference_transforms = A.Compose([
        ToTensorV2(),
    ])
    train_ds = BDDSegmentationDataset(train_df, transform=train_transforms)
    val_ds = BDDSegmentationDataset(val_df, transform=inference_transforms)
    test_ds = BDDSegmentationDataset(test_df, transform=inference_transforms)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=Configuration.BATCH_SIZE,
        shuffle=Configuration.APPLY_SHUFFLE
    )
    val_loader = DataLoader(
            dataset=val_ds,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=Configuration.APPLY_SHUFFLE
        )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=Configuration.BATCH_SIZE,
        shuffle=Configuration.APPLY_SHUFFLE
    )

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=Configuration.CHANNELS,
        classes=Configuration.NUM_CLASSES
    )

    print(
        summary(
                model=model,
                input_size=(Configuration.BATCH_SIZE, Configuration.CHANNELS, Configuration.IMAGE_HEIGHT, Configuration.IMAGE_WIDTH),
                col_names=["output_size", "num_params", "trainable"],
                col_width=30,
                row_settings=["var_names"],
                depth=5
            )
    )

    # Define Loss Function
    # Dice loss operates on (B, NUM_CLASSES, H, W) logits vs one-hot masks,
    # which matches the multi-class semantic segmentation task. CrossEntropyLoss
    # would require integer class-id targets of shape (B, H, W).
    loss_fn = DiceLoss()

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Configuration.LR
    )

    # Define Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=Configuration.PATIENCE
    )

    print('Training U-Net Model')
    print(f'Train on {len(train_df)} samples, validate on {len(val_df)} samples.')
    print('----------------------------------')

    # Generate training session config
    session_config = {
        'model'               : model,
        'train_dataloader'    : train_loader,
        'eval_dataloader'     : val_loader,
        'optimizer'           : optimizer,
        'scheduler'           : scheduler,
        'loss_fn'             : loss_fn,
        'epochs'              : Configuration.EPOCHS,
        'device'              : Configuration.DEVICE
    }

    # Execute Training Session
    unet_session_history = train(**session_config)

    # Create Model directory
    model_name = 'teacher'
    model_path = './model/'
    os.mkdir(model_path)

    # Save Model
    torch.save(model, model_path + model_name + '.pt')

    # Convert U-Net history dict to DataFrame
    unet_session_history_df = pd.DataFrame(unet_session_history)
    print(unet_session_history_df[:5])

    # Plot U-Net Session Training History
    plot_training_curves(
        unet_session_history,
        fig_size=(20, 20)
    )

    # Generate Segmentation Metrics
    unet_metrics = compute_metrics(
        model, test_loader, Configuration.DEVICE
    )

    # Create copy of test df
    unet_test_df = test_df.copy()

    # Concatenate Metrics onto copied df
    unet_test_df = pd.concat(
        (unet_test_df, pd.DataFrame(unet_metrics)),
        axis=1
    )

    # View df
    print(unet_test_df[:5])

    # Export a grid of random test samples (image / image+true mask /
    # image+predicted mask) so the model output can be inspected visually.
    visualize_predictions(
        model,
        test_df,
        torch.device(Configuration.DEVICE)
    )


if __name__ == "__main__":
    main()
