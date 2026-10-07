import gc
import glob
import os
from typing import Dict, List, cast

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, Int64, UInt8, jaxtyped
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchinfo import summary
from tqdm import tqdm

# Shape aliases that document the tensor layout at each stage of the pipeline.
#
# ``h``/``w`` are the spatial dimensions, ``b`` is the batch size and ``c`` is
# the number of channels/classes. These aliases are enforced at runtime by
# ``jaxtyped`` + ``beartype``, so a shape mismatch raises a ``TypeCheckError``
# instead of a confusing downstream broadcast error.
ImageTensor = Float[Tensor, "3 h w"]  # single image, channel-first layout
IndexMaskTensor = Int64[Tensor, "h w"]  # single class-index mask
BatchImage = Float[Tensor, "b 3 h w"]  # collated batch of images
BatchIndexMask = Int64[Tensor, "b h w"]  # collated batch of class-index masks
Logits = Float[Tensor, "b c h w"]  # model output, ``c == NUM_CLASSES``
Scalar = Float[Tensor, ""]  # scalar (0-dim) tensor
NumpyImage = Float[np.ndarray, "h w 3"]  # single image, channels-last layout
ClassIndexArray = UInt8[np.ndarray, "h w"]  # per-pixel class id map (numpy)
RawColorImage = UInt8[np.ndarray, "h w 3"]  # single image, channels-last uint8 layout


# BDD100k color-label palette. The row index is the class id (0-19), matching
# ``Configuration.NUM_CLASSES``. Predicted class maps are coloured with this
# same palette so they render side by side with the ground-truth color labels
# stored on disk. The exact colour per class only needs to be distinct and
# consistent; it mirrors the default BDD100k colours.
CLASS_COLORS: np.ndarray = np.array(
    [
        [128, 64, 128],  # 0  - Road
        [244, 35, 232],  # 1  - Sidewalk
        [70, 70, 70],  # 2  - Building
        [102, 102, 156],  # 3  - Wall
        [190, 153, 153],  # 4  - Fence
        [153, 153, 153],  # 5  - Pole
        [250, 170, 30],  # 6  - Traffic Light
        [220, 220, 0],  # 7  - Traffic Sign
        [107, 142, 35],  # 8  - Vegetation
        [152, 251, 152],  # 9  - Terrain
        [70, 130, 180],  # 10 - Sky
        [220, 20, 60],  # 11 - Person
        [255, 0, 0],  # 12 - Rider
        [0, 0, 142],  # 13 - Car
        [0, 0, 70],  # 14 - Truck
        [0, 60, 100],  # 15 - Bus
        [0, 80, 100],  # 16 - Train
        [0, 0, 230],  # 17 - Motorcycle
        [119, 11, 32],  # 18 - Bicycle
        [0, 0, 0],  # 19 - Unknown
    ],
    dtype=np.uint8,
)

# Human-readable names for each palette row, kept in lock-step with
# ``CLASS_COLORS`` so class ids can be logged without consulting the palette
# comments by hand.
CLASS_NAMES: list[str] = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "unknown",
]


@jaxtyped(typechecker=beartype)
def color_label_to_class_index(label: RawColorImage) -> ClassIndexArray:
    """Map an RGB color-label image to a per-pixel class-index map.

    BDD100k stores segmentation masks as RGB PNGs whose colours are exactly the
    entries of ``CLASS_COLORS``. Semantic segmentation needs the class id per
    pixel (shape ``(H, W)``) rather than the RGB representation (shape
    ``(H, W, 3)``), so this conversion must happen before the mask is turned
    into a tensor.

    Instead of allocating one boolean mask per palette colour, the RGB triplets
    are bit-packed into single 32-bit integer keys. This turns the per-pixel
    colour matching into single comparison instructions and avoids creating 20
    intermediate boolean arrays for every image.

    Steps
    -----
    1. Initialise the output with the id of the last palette entry so that any
       unknown colour degrades to ``Unknown`` instead of producing an invalid
       index.
    2. Pack the image's RGB channels into a single 32-bit integer ``(R << 16) |
       (G << 8) | B`` and do the same for every palette colour.
    3. For each packed palette key, assign the corresponding class id to every
       pixel whose packed key matches.

    Parameters
    ----------
    label : RawColorImage
        RGB color-label array of shape ``(H, W, 3)`` and dtype ``uint8``.

    Returns
    -------
    ClassIndexArray
        Class-index array of shape ``(H, W)`` and dtype ``uint8``, whose values
        are in ``[0, len(CLASS_COLORS))``.
    """
    if label.ndim != 3 or label.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {label.shape}.")

    # Default to the last class id so unknown colours fall back gracefully
    # instead of indexing the palette out of bounds later.
    class_ids = np.full(label.shape[:2], len(CLASS_COLORS) - 1, dtype=np.uint8)

    # Pack RGB channels into a single int32 scalar so a class match reduces to
    # one scalar equality check instead of three per-colour channel comparisons.
    label_packed = (
        (label[..., 0].astype(np.int32) << 16)
        | (label[..., 1].astype(np.int32) << 8)
        | label[..., 2].astype(np.int32)
    )
    palette_packed = (
        (CLASS_COLORS[:, 0].astype(np.int32) << 16)
        | (CLASS_COLORS[:, 1].astype(np.int32) << 8)
        | CLASS_COLORS[:, 2].astype(np.int32)
    )

    # Assign class indices where the packed pixel keys match the palette keys.
    for class_id, color_key in enumerate(palette_packed):
        class_ids[label_packed == color_key] = class_id

    return class_ids


class Configuration:
    """Global configuration settings for data loading, training, and validation."""

    DEVICE: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    NUM_DEVICES: int = torch.cuda.device_count()
    # Two background workers overlap disk I/O with GPU execution.
    NUM_WORKERS: int = 2

    NUM_CLASSES: int = 20
    EPOCHS: int = 20
    # Total batch size across both GPUs (8 per GPU keeps VRAM stable without OOM).
    BATCH_SIZE: int = 16 if torch.cuda.device_count() >= 2 else 8
    LR: float = 1e-4
    PATIENCE: int = 8

    APPLY_SHUFFLE: bool = True
    SEED: int = 768

    # Spatial dimensions of raw BDD100k images before preprocessing.
    ORIGINAL_IMAGE_HEIGHT: int = 720
    ORIGINAL_IMAGE_WIDTH: int = 1280

    # Downsampled resolution balancing spatial detail against GPU memory.
    IMAGE_HEIGHT: int = 360
    IMAGE_WIDTH: int = 640
    CHANNELS: int = 3  # Standard RGB channels.


class Path:
    """Filesystem path definitions for datasets, masks, and precomputed weights."""

    BASE: str = "./data/bdd100k"

    SEGMENTATION_MASK_LABEL_FOLDER: str = BASE + "/segmentation_maps/color_labels"
    SEGMENTATION_MASK_TRAIN_PATH: str = SEGMENTATION_MASK_LABEL_FOLDER + "/train"
    SEGMENTATION_MASK_VAL_PATH: str = SEGMENTATION_MASK_LABEL_FOLDER + "/val"

    IMAGE_FOLDER: str = BASE + "/images_10k"
    IMAGE_TRAIN_PATH: str = IMAGE_FOLDER + "/train"
    IMAGE_VAL_PATH: str = IMAGE_FOLDER + "/val"

    SAMPLE_WEIGHTS_PATH: str = "./data/sample_weights.npy"


class BDDSegmentationDataset(Dataset[tuple[ImageTensor, IndexMaskTensor]]):
    """Dataset loader for BDD100k semantic segmentation.

    Reads paired camera frames and segmentation maps from disk, applies spatial
    and photometric augmentations, and formats tensors for PyTorch model
    ingestion. Masks are kept as compact 2-D ``int64`` class-index maps rather
    than one-hot ``float`` tensors to conserve host memory.
    """

    def __init__(self, df: pd.DataFrame, transform: A.Compose | None = None) -> None:
        """Initialise file references and the transformation pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ``image_paths`` and ``mask_paths`` columns.
        transform : A.Compose | None, optional
            Albumentations transformation pipeline. Defaults to None.
        """
        super().__init__()

        # Materialise pandas columns to Python lists, avoiding pandas indexing
        # overhead on every ``__getitem__`` call.
        self.image_paths: list[str] = df["image_paths"].to_list()
        self.mask_paths: list[str] = df["mask_paths"].to_list()
        self.transform: A.Compose | None = transform

    @jaxtyped(typechecker=beartype)
    def load_sample(self, index: int) -> tuple[NumpyImage, ClassIndexArray]:
        """Load the image and its per-pixel class-index mask at ``index``.

        The image is opened as RGB and normalised to ``[0, 1]``. The mask PNG is
        likewise forced to RGB (some BDD100k color-labels carry an alpha
        channel) before being collapsed from the RGB color-label format to a
        single class id per pixel via :func:`color_label_to_class_index`.

        Steps
        -----
        1. Validate ``index`` against the dataset bounds.
        2. Open the image and normalise its pixels to ``[0, 1]``.
        3. Open the mask, force RGB, and map its colours to class indices.
        4. Use context managers so file descriptors are released immediately.

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
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(
                f"Index {index} is out of bounds for dataset of length {len(self)}."
            )

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Force RGB so that RGBA sources (e.g. some BDD100k color-label PNGs
        # carry an alpha channel) are reduced to 3 channels.
        with Image.open(image_path) as image_pil:
            image = np.array(image_pil.convert("RGB"), dtype=np.float32) / 255.0

        with Image.open(mask_path) as mask_pil:
            raw_mask = np.array(mask_pil.convert("RGB"), dtype=np.uint8)
            class_mask = color_label_to_class_index(raw_mask)

        return image, class_mask

    def __len__(self) -> int:
        """Return the total number of samples present in the dataset.

        Returns
        -------
        int
            Length of the underlying sample list.
        """
        return len(self.image_paths)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> tuple[ImageTensor, IndexMaskTensor]:
        """Retrieve and transform the sample at position ``index``.

        ``ToTensorV2`` converts the image from ``(H, W, 3)`` to ``(3, H, W)``
        and leaves the 2-D class-index mask as ``(H, W)``. The mask is then
        cast to ``int64``; one-hot encoding is deferred to the loss functions so
        the dataloader never materialises a 20-channel float tensor, which would
        otherwise inflate host memory 20x and risk OOM during collation.

        Steps
        -----
        1. Fetch the raw image array and 2-D integer class mask via ``load_sample``.
        2. Apply Albumentations or the default channel transposition.
        3. Cast the transformed mask to ``int64`` and return both tensors.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[ImageTensor, IndexMaskTensor]
            Transformed image tensor ``(3, H, W)`` and integer mask tensor ``(H, W)``.
        """
        image, class_mask = self.load_sample(index)

        # Apply Albumentations pipeline or default channel transposition. The
        # mask is a 2-D class map, so no channel transposition is needed for it.
        if self.transform:
            transformed = self.transform(image=image, mask=class_mask)
        else:
            transformed = ToTensorV2()(image=image, mask=class_mask)

        # Keep the mask as a compact (H, W) int64 tensor instead of expanding it
        # to a one-hot float tensor.
        mask_tensor = transformed["mask"].to(torch.int64)

        return transformed["image"], mask_tensor


def find_image_path_from_mask(complete_mask_path: str, base_image_path: str) -> str:
    file_path_split = complete_mask_path.split("/")
    mask_file_name = file_path_split[-1].split("_")[0]

    image_path = base_image_path + "/" + mask_file_name + ".jpg"
    return image_path


def find_train_image_path_from_mask(complete_mask_path: str) -> str:
    return find_image_path_from_mask(complete_mask_path, Path.IMAGE_TRAIN_PATH)


def find_val_image_path_from_mask(complete_mask_path: str) -> str:
    return find_image_path_from_mask(complete_mask_path, Path.IMAGE_VAL_PATH)


def find_mask_path_from_image(complete_image_path: str, base_mask_path: str) -> str:
    file_path_split = complete_image_path.split("/")
    mask_file_name = file_path_split[-1].split(".")[0]

    mask_path = base_mask_path + "/" + mask_file_name + "_train_color.png"
    return mask_path


def find_train_mask_path_from_image(complete_image_path: str) -> str:
    return find_mask_path_from_image(
        complete_image_path, Path.SEGMENTATION_MASK_TRAIN_PATH
    )


def find_val_mask_path_from_image(complete_image_path: str) -> str:
    return find_mask_path_from_image(
        complete_image_path, Path.SEGMENTATION_MASK_VAL_PATH
    )


def load_dataset_from_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # load train, then split into train-test
    train_mask_paths = glob.glob(f"{Path.SEGMENTATION_MASK_TRAIN_PATH}/*.png")
    problematic_masks = []
    for complete_mask_path in train_mask_paths:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_masks.append(complete_mask_path)
                train_mask_paths.remove(complete_mask_path)
    print(f"Problematic masks: {problematic_masks}")

    train_image_paths = list(map(find_train_image_path_from_mask, train_mask_paths))
    problematic_images = []
    for complete_image_path in train_image_paths:
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_images.append(complete_image_path)
                train_image_paths.remove(complete_image_path)
                train_mask_paths.remove(
                    find_train_mask_path_from_image(complete_image_path)
                )
    print(f"Problematic images: {problematic_images}")

    train_test_df = pd.DataFrame(
        {
            "image_paths": train_image_paths,
            "mask_paths": train_mask_paths,
        }
    )
    train_df, test_df = train_test_split(
        train_test_df, test_size=0.2, random_state=Configuration.SEED
    )

    # load val
    val_mask_paths = glob.glob(f"{Path.SEGMENTATION_MASK_VAL_PATH}/*.png")
    problematic_val_masks = []
    for complete_mask_path in val_mask_paths:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_val_masks.append(complete_mask_path)
                val_mask_paths.remove(complete_mask_path)
    print(f"Problematic val masks: {problematic_val_masks}")

    val_image_paths = list(map(find_val_image_path_from_mask, val_mask_paths))
    problematic_val_images = []
    for complete_image_path in val_image_paths:
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_val_images.append(complete_image_path)
                val_image_paths.remove(complete_image_path)
                val_mask_paths.remove(
                    find_val_mask_path_from_image(complete_image_path)
                )
    print(f"Problematic val images: {problematic_val_images}")

    val_df = pd.DataFrame(
        {
            "image_paths": val_image_paths,
            "mask_paths": val_mask_paths,
        }
    )

    return train_df, val_df, test_df


@jaxtyped(typechecker=beartype)
def forward(model: nn.Module, x: BatchImage) -> Logits:
    """Run a single forward pass and assert the input/output shapes.

    Centralising the forward pass here lets ``jaxtyping`` verify that the input
    batch is always ``(B, 3, H, W)`` and that the model produces
    ``(B, NUM_CLASSES, H, W)`` logits, which is where shape confusion most often
    arises.

    Steps
    -----
    1. Run the input tensor batch through the model.
    2. Return the raw logits for downstream loss computation.

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


class CompoundLoss(nn.Module):
    """Compound loss combining Dice Loss and Focal Loss for semantic segmentation.

    Semantic segmentation on class-imbalanced datasets benefits from combining a
    region-based loss (Dice loss) and a distribution-based loss (Focal loss).
    Dice loss optimizes overall mask overlap to handle class imbalance, while
    Focal loss down-weights easy background pixels to focus gradient updates on
    hard boundaries and minority classes.

    Steps
    -----
    1. Initialize the underlying SMP DiceLoss and FocalLoss modules in multiclass mode.
    2. In the forward pass, inspect target tensor dimensionality; if 4D (one-hot encoded),
       collapse the channel dimension to 2D class indices via ``argmax(dim=1)``.
    3. Compute multi-class Dice loss from raw logits.
    4. Compute multi-class Focal loss from raw logits.
    5. Return the weighted sum: ``dice_weight * dice + focal_weight * focal``.

    Parameters
    ----------
    dice_weight : float, optional
        Weight factor applied to the Dice loss component. Defaults to 0.5.
    focal_weight : float, optional
        Weight factor applied to the Focal loss component. Defaults to 1.0.
    ignore_index : int, optional
        Class index to ignore during loss computation. Defaults to 19.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 1.0,
        ignore_index: int = 19,
    ) -> None:
        super().__init__()
        # Store loss weighting factors to adjust the relative contribution of each loss term
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # SMP DiceLoss operates directly on raw logits when from_logits=True,
        # avoiding an explicit external softmax step
        self.dice_loss = smp.losses.DiceLoss(
            mode=smp.losses.MULTICLASS_MODE,
            from_logits=True,
            ignore_index=ignore_index,
        )

        # SMP FocalLoss handles multiclass cross-entropy while masking out
        # the designated unknown/ignored class id
        self.focal_loss = smp.losses.FocalLoss(
            mode=smp.losses.MULTICLASS_MODE,
            ignore_index=ignore_index,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the weighted compound loss between predictions and targets.

        Steps
        -----
        1. Check if targets are 4D (one-hot encoded); if so, collapse to class indices.
        2. Evaluate Dice loss and Focal loss independently.
        3. Weight and sum both loss terms.

        Parameters
        ----------
        logits : Tensor
            Raw unnormalized model predictions of shape ``(B, C, H, W)``.
        targets : Tensor
            Ground-truth masks, either one-hot encoded tensors of shape ``(B, C, H, W)``
            or class indices of shape ``(B, H, W)``.

        Returns
        -------
        Tensor
            Scalar compound loss tensor suitable for gradient backpropagation.
        """
        # Collapse one-hot targets (B, C, H, W) to class indices (B, H, W) because
        # SMP multiclass loss functions with ignore_index require integer class indices
        if targets.ndim == 4:
            targets = targets.argmax(dim=1)

        dice = cast(Scalar, self.dice_loss(logits, targets))
        focal = cast(Scalar, self.focal_loss(logits, targets))

        # Combine weighted losses to balance boundary refinement and region overlap
        return self.dice_weight * dice + self.focal_weight * focal


@jaxtyped(typechecker=beartype)
def kl_divergence(probs: Logits, adv_probs: Logits, eps: float = 1e-8) -> Scalar:
    """Compute mean per-pixel Kullback-Leibler divergence ``KL(p || q)``.

    TRADES measures robustness as the divergence between the model's clean
    prediction ``p`` and its prediction on a perturbed input ``q``. Because a
    segmentation model outputs a per-pixel class distribution, the KL divergence
    is evaluated independently at every pixel and then averaged over the batch
    and spatial dimensions, so every pixel contributes equally to the regulariser
    regardless of the image resolution.

    Steps
    -----
    1. Add ``eps`` to both distributions so the logarithm never sees a zero
       probability (which would otherwise produce ``-inf``).
    2. Sum ``p * (log p - log q)`` over the class axis, yielding one KL value
       per pixel of shape ``(B, H, W)``.
    3. Average those per-pixel KL values into a single scalar.

    Parameters
    ----------
    probs : Logits
        Clean class probabilities of shape ``(B, C, H, W)``.
    adv_probs : Logits
        Adversarial class probabilities of shape ``(B, C, H, W)``.
    eps : float, optional
        Smoothing term added before the logarithm to avoid ``log(0)``.
        Defaults to ``1e-8``.

    Returns
    -------
    Scalar
        Mean KL divergence, differentiable w.r.t. ``adv_probs``.
    """
    # ``eps`` keeps the log well-defined when a class probability is exactly
    # zero after softmax (which happens for confident, wrong predictions).
    kl_per_pixel = (probs * (torch.log(probs + eps) - torch.log(adv_probs + eps))).sum(
        dim=1
    )

    # Average over batch and spatial axes so the regulariser is resolution
    # independent and directly comparable to the natural loss magnitude.
    return kl_per_pixel.mean()


class TradesLoss(nn.Module):
    """TRADES adversarial loss for semantic segmentation.

    Combines a natural (clean) loss with a KL-divergence robustness term, as
    proposed by Zhang et al. (2019) "Theoretically Principled Trade-off between
    Robustness and Accuracy". The natural loss encourages accurate clean
    predictions, while the robustness term pushes the model to keep its output
    stable for adversarial inputs inside an ``epsilon`` ball around each sample.

    The adversarial input is crafted with projected gradient descent (PGD): for
    ``num_steps`` iterations the perturbation is moved by ``alpha`` in the
    direction of the signed gradient of ``KL(clean || adversarial)`` and then
    projected back onto the ``L_inf`` ball of radius ``epsilon``. The final
    loss is ``natural_loss(clean) + beta * KL(clean || adversarial)``.

    Parameters
    ----------
    natural_loss : nn.Module
        Loss applied to clean logits, e.g. :class:`CompoundLoss`.
    epsilon : float, optional
        Radius of the ``L_inf`` adversarial perturbation ball. Defaults to 0.03.
    alpha : float, optional
        PGD step size. Defaults to 0.01.
    num_steps : int, optional
        Number of PGD iterations used to craft the perturbation. Defaults to 10.
    beta : float, optional
        Weight of the robustness term relative to the natural loss. Defaults to
        6.0.

    Notes
    -----
    The ``model`` is passed to :meth:`forward` rather than stored on the
    instance, so the loss object stays independent of any particular network
    checkpoint and can be reused across runs.
    """

    def __init__(
        self,
        natural_loss: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 10,
        beta: float = 6.0,
    ) -> None:
        """Configure TRADES perturbation boundaries and weighting hyperparameters.

        Parameters
        ----------
        natural_loss : nn.Module
            Supervised criterion applied to clean model logits.
        epsilon : float, optional
            L-infinity perturbation boundary radius. Defaults to 0.03.
        alpha : float, optional
            Single step perturbation increment. Defaults to 0.01.
        num_steps : int, optional
            Number of PGD iterations per batch. Defaults to 10.
        beta : float, optional
            Trade-off weight balancing natural loss against robustness. Defaults to 6.0.
        """
        super().__init__()
        self.natural_loss: nn.Module = natural_loss
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.num_steps: int = num_steps
        self.beta: float = beta

    def forward(
        self, model: nn.Module, x: BatchImage, y: BatchIndexMask
    ) -> tuple[Scalar, Logits]:
        """Compute natural loss, craft a PGD adversary, and evaluate robustness loss.

        Steps
        -----
        1. Evaluate the clean forward pass, compute the natural loss, and detach
           the clean probabilities.
        2. Switch the model to eval mode during PGD so BatchNorm running
           statistics are not polluted by the intermediate adversarial steps.
        3. Iteratively ascend the KL divergence gradient while projecting back
           onto the epsilon ``L_inf`` ball and the ``[0, 1]`` pixel range.
        4. Restore the original training state and evaluate the final robustness
           regulariser on the adversarial image.
        5. Return the total loss plus the detached clean logits so the caller
           does not need a redundant forward pass for metric tracking.

        Parameters
        ----------
        model : nn.Module
            Segmentation network being optimised.
        x : BatchImage
            Clean image batch of shape ``(B, 3, H, W)``.
        y : BatchIndexMask
            Target class-index batch of shape ``(B, H, W)``.

        Returns
        -------
        tuple[Scalar, Logits]
            Tuple containing the differentiable composite loss and the detached
            clean logits.
        """
        # 1. Clean branch: baseline predictions and the task loss. The clean
        # probabilities are detached so the adversarial steps only propagate
        # gradients through the perturbed branch.
        clean_logits = forward(model, x)
        natural = cast(Scalar, self.natural_loss(clean_logits, y))
        clean_probs = torch.softmax(clean_logits, dim=1).detach()

        # 2. PGD attack while freezing BatchNorm updates. Running BatchNorm in
        # training mode over ``num_steps`` iterations would corrupt the moving
        # mean/variance statistics and retain unnecessary graphs.
        was_training = model.training
        model.eval()

        x_adv = x.clone().detach()
        for _ in range(self.num_steps):
            # Re-enable autograd on the current adversarial image so the step
            # walks the loss surface of the robustness term.
            x_adv.requires_grad_(True)

            adv_logits = forward(model, x_adv)
            adv_probs = torch.softmax(adv_logits, dim=1)
            kl = kl_divergence(clean_probs, adv_probs)

            # dKL/dx_adv; stepping along its sign is the standard PGD update.
            grad = torch.autograd.grad(kl, x_adv)[0]

            # Apply the step, then project onto the intersection of the L_inf
            # ball and the valid [0, 1] pixel range.
            perturbation = x_adv.detach() + self.alpha * grad.sign() - x
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + perturbation, 0.0, 1.0).detach()

        # Restore the original model mode before the final adversarial pass.
        if was_training:
            model.train()

        # 3. Robustness term on the final adversarial example: penalise how much
        # the perturbation moved the prediction away from the clean prediction.
        adv_logits = forward(model, x_adv)
        adv_probs = torch.softmax(adv_logits, dim=1)
        robust = kl_divergence(clean_probs, adv_probs)

        total_loss = natural + self.beta * robust

        # Return the clean logits (detached) alongside the loss to avoid a
        # duplicate forward pass in the training loop.
        return total_loss, clean_logits.detach()


@torch.no_grad()
def compute_batch_macro_iou(
    y_pred: torch.Tensor,  # (B, C, H, W) logits
    y_true: torch.Tensor,  # (B, H, W) class indices
    num_classes: int = 19,  # Classes 0 to 18 (excludes 19: Unknown)
    eps: float = 1e-7,
) -> float:
    """Compute the mean macro IoU over a batch, excluding ignored classes.

    The model outputs raw logits while the ground truth is a class-index map, so
    the logits are reduced to a single class id per pixel via ``argmax``. IoU is
    then computed class by class and averaged only over the classes that
    actually occur in either the prediction or the ground truth; empty classes
    are skipped rather than counted as zero, which would otherwise drag the mean
    down on batches dominated by a few classes.

    Steps
    -----
    1. Collapse logits to ``(B, H, W)`` class-index maps via ``argmax``.
    2. For each class in ``[0, num_classes)``, compute intersection over union.
    3. Average the IoU of every class whose union is non-zero.

    Parameters
    ----------
    y_pred : torch.Tensor
        Model logits of shape ``(B, C, H, W)``.
    y_true : torch.Tensor
        Class-index targets of shape ``(B, H, W)``.
    num_classes : int, optional
        Number of classes to score, excluding the ``Unknown`` class. Defaults
        to ``19`` (classes 0-18).
    eps : float, optional
        Smoothing constant added to intersection and union to avoid division by
        zero. Defaults to ``1e-7``.

    Returns
    -------
    float
        Mean IoU over the classes present in the batch, or ``0.0`` if no class
        has a non-empty union.
    """
    preds = y_pred.argmax(dim=1)  # (B, H, W)

    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        true_mask = y_true == cls

        intersection = (pred_mask & true_mask).sum().float().item()
        union = (pred_mask | true_mask).sum().float().item()

        # Only include the class in the mean if it exists in GT or Prediction
        if union > 0:
            iou_per_class.append((intersection + eps) / (union + eps))

    return float(np.mean(iou_per_class)) if iou_per_class else 0.0


@jaxtyped(typechecker=beartype)
def execute_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[ImageTensor, IndexMaskTensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: TradesLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run one adversarial training epoch and return its average metrics.

    Each batch is passed to :class:`TradesLoss`, which produces the natural loss
    plus the TRADES KL robustness term. Gradient descent is applied to that
    combined loss, while the per-batch macro IoU on the detached clean logits is
    accumulated as a quality metric alongside the loss.

    Steps
    -----
    1. Set the model to training mode and initialise metric accumulators.
    2. Iterate over the dataloader, transferring tensors to the device.
    3. Zero gradients, evaluate TRADES loss, backpropagate, and step weights.
    4. Accumulate loss and macro IoU from the returned clean logits.
    5. Return epoch-level average loss and IoU.

    Parameters
    ----------
    model : nn.Module
        Segmentation model being trained.
    dataloader : DataLoader[tuple[ImageTensor, IndexMaskTensor]]
        Batched training data.
    optimizer : torch.optim.Optimizer
        Optimiser used for the parameter update.
    loss_fn : TradesLoss
        Adversarial TRADES loss that returns the loss and detached clean logits.
    device : torch.device
        Device the training runs on.

    Returns
    -------
    tuple[float, float]
        Mean training loss and mean macro IoU over the epoch.
    """
    # Set model into training mode
    model.train()

    # Initialize train loss & accuracy
    train_loss, train_iou = 0.0, 0.0

    # Execute training loop over train dataloader
    for X, y in dataloader:
        # Load data onto target device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Reset Gradients
        optimizer.zero_grad()

        # TRADES loss returns the total loss and detached clean logits so the
        # IoU metric can be computed without a redundant forward pass.
        loss, clean_logits = loss_fn(model, X, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iou += compute_batch_macro_iou(clean_logits, y, num_classes=19)

    # Compute Step Metrics
    train_loss = train_loss / len(dataloader)
    train_iou = train_iou / len(dataloader)

    return train_loss, train_iou


@jaxtyped(typechecker=beartype)
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[ImageTensor, IndexMaskTensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a dataloader and return mean loss and macro IoU.

    The model is placed in eval mode and run under ``torch.inference_mode`` so
    that no gradients are tracked. Each batch's clean loss and macro IoU are
    accumulated and normalised by the number of batches.

    Steps
    -----
    1. Set the model to eval mode and enter ``torch.inference_mode``.
    2. Pass clean batches through the model without gradient tracking.
    3. Accumulate validation loss and macro IoU.
    4. Return normalised validation metrics.

    Parameters
    ----------
    model : nn.Module
        Segmentation model being evaluated.
    dataloader : DataLoader[tuple[ImageTensor, IndexMaskTensor]]
        Batched validation data.
    loss_fn : nn.Module
        Clean loss callable that consumes ``(logits, class-index targets)``.
    device : torch.device
        Device the evaluation runs on.

    Returns
    -------
    tuple[float, float]
        Mean evaluation loss and mean macro IoU over the epoch.
    """
    # Set model into eval mode
    model.eval()

    # Initialize eval loss & accuracy
    eval_loss, eval_iou = 0.0, 0.0

    # Active inference context manager
    with torch.inference_mode():
        # Execute eval loop over dataloader
        for X, y in dataloader:
            # Load data onto target device
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Feed-forward and compute metrics
            y_pred = forward(model, X)
            loss = loss_fn(y_pred, y)
            eval_loss += loss.item()

            # Compute Macro IoU for the batch (excluding class 19)
            eval_iou += compute_batch_macro_iou(y_pred, y, num_classes=19)

    # Compute Step Metrics
    eval_loss = eval_loss / len(dataloader)
    eval_iou = eval_iou / len(dataloader)

    return eval_loss, eval_iou


@jaxtyped(typechecker=beartype)
def train(
    model: nn.Module,
    train_dataloader: DataLoader[tuple[ImageTensor, IndexMaskTensor]],
    eval_dataloader: DataLoader[tuple[ImageTensor, IndexMaskTensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: TradesLoss,
    eval_loss_fn: nn.Module,
    epochs: int,
    train_device: torch.device,
    eval_device: torch.device,
    scheduler: lr_scheduler.ReduceLROnPlateau | None = None,
) -> tuple[nn.Module, Dict[str, List[float]]]:
    """Run full TRADES training with periodic clean validation.

    Training epochs use the adversarial :class:`TradesLoss`, while validation is
    measured with the clean ``eval_loss_fn`` (no adversary) so the reported
    evaluation metrics reflect real-world, unperturbed performance. The best
    checkpoint is retained by lowest validation loss and restored before
    returning.

    Steps
    -----
    1. Initialise the metric history and checkpoint-tracking variables.
    2. For each epoch, execute adversarial training and clean validation.
    3. Save an unwrapped CPU state-dict copy when validation loss improves.
    4. Step the learning-rate scheduler if provided.
    5. Release host/device memory at epoch end to prevent fragmentation.
    6. Restore the best checkpoint before returning.

    Parameters
    ----------
    model : nn.Module
        Segmentation model being trained.
    train_dataloader : DataLoader[tuple[ImageTensor, IndexMaskTensor]]
        Batched training data.
    eval_dataloader : DataLoader[tuple[ImageTensor, IndexMaskTensor]]
        Batched validation data.
    optimizer : torch.optim.Optimizer
        Optimiser used for the parameter update.
    loss_fn : TradesLoss
        Adversarial loss used for gradient updates during training.
    eval_loss_fn : nn.Module
        Clean loss used to compute the validation loss.
    epochs : int
        Number of passes over the training data.
    train_device : torch.device
        Device used for training.
    eval_device : torch.device
        Device used for validation.
    scheduler : lr_scheduler.ReduceLROnPlateau | None, optional
        Optional learning-rate scheduler stepped on the validation loss.
        Defaults to None.

    Returns
    -------
    tuple[nn.Module, Dict[str, List[float]]]
        The unwrapped model with the best checkpoint restored, and the per-epoch
        metric history of training/eval loss and macro IoU.
    """
    # Initialize training session
    session: Dict[str, List[float]] = {
        "loss": [],
        "macro_iou_score": [],
        "eval_loss": [],
        "eval_macro_iou_score": [],
    }

    # Track the checkpoint with the lowest validation loss so the final model
    # can be reverted to the best-seen weights instead of the last epoch's.
    best_eval_loss = float("inf")
    best_model_state: Dict[str, Tensor] | None = None

    # Training loop
    for epoch in tqdm(range(epochs)):
        # Execute Epoch
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_iou = execute_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            train_device,
        )

        # Evaluate Model
        eval_loss, eval_iou = evaluate(
            model,
            eval_dataloader,
            eval_loss_fn,
            eval_device,
        )

        # Access the raw unwrapped model so saved weights are agnostic of DataParallel
        raw_model = model.module if isinstance(model, nn.DataParallel) else model

        # Keep a snapshot whenever the validation loss improves so the best
        # checkpoint is available for the final test evaluation.
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_state = {
                name: param.detach().cpu().clone()
                for name, param in raw_model.state_dict().items()
            }

        # Execute scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(eval_loss)
            current_lr = optimizer.param_groups[0]["lr"]

        # Log Epoch Metrics
        log_text = (
            f"loss: {train_loss:.4f} - train_macro_iou: {train_iou:.4f} - "
            f"eval_loss: {eval_loss:.4f} - eval_macro_iou_score: {eval_iou:.4f}"
        )

        if scheduler:
            print(log_text + f" - lr: {current_lr}")
        else:
            print(log_text)

        # Record Epoch Metrics
        session["loss"].append(train_loss)
        session["macro_iou_score"].append(train_iou)
        session["eval_loss"].append(eval_loss)
        session["eval_macro_iou_score"].append(eval_iou)

        # Explicitly invoke garbage collection and release cached PyTorch memory.
        # This prevents fragmented tensors from steadily accumulating in host memory across epochs.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Restore the best checkpoint so the model returned to the caller (and the
    # one evaluated on the test set downstream) reflects the lowest eval loss.
    if best_model_state is not None:
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(best_model_state)

    # Return raw model and session metrics
    return raw_model, session


def plot_training_curves(
    history: Dict[str, List[float]],
    fig_size: tuple[int, int] = (20, 10),
) -> None:

    loss = np.array(history["loss"])
    val_loss = np.array(history["eval_loss"])

    iou = np.array(history["macro_iou_score"])
    val_iou = np.array(history["eval_macro_iou_score"])

    epochs = range(len(history["loss"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)

    # Plot loss
    ax1.plot(epochs, loss, label="training_loss", marker="o", color="C5")
    ax1.plot(epochs, val_loss, label="eval_loss", marker="o", color="C6")

    # Fill area between losses
    ax1.fill_between(
        epochs,
        loss,
        val_loss,
        where=(loss > val_loss),
        color="C5",
        alpha=0.4,
        interpolate=True,
    )
    ax1.fill_between(
        epochs,
        loss,
        val_loss,
        where=(loss < val_loss),
        color="C6",
        alpha=0.4,
        interpolate=True,
    )

    # Add Text & Formats
    ax1.set_title("Loss (Lower Means Better)", fontsize=22)
    ax1.set_xlabel("Epochs", fontsize=18)
    ax1.set_ylabel("Loss", fontsize=18)
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax1.legend(fontsize=14)

    # Plot metric
    ax2.plot(epochs, iou, label="training_macro_iou", marker="o", color="C5")
    ax2.plot(epochs, val_iou, label="eval_macro_iou", marker="o", color="C6")

    # Fill area between metrics
    ax2.fill_between(
        epochs,
        iou,
        val_iou,
        where=(iou > val_iou),
        color="C5",
        alpha=0.4,
        interpolate=True,
    )
    ax2.fill_between(
        epochs,
        iou,
        val_iou,
        where=(iou < val_iou),
        color="C6",
        alpha=0.4,
        interpolate=True,
    )

    # Add Text & Formats
    ax2.set_title("Macro IoU (Higher Means Better)", fontsize=22)
    ax2.set_xlabel("Epochs", fontsize=18)
    ax2.set_ylabel("Macro IoU", fontsize=18)
    ax2.tick_params(axis="both", which="major", labelsize=14)
    ax2.legend(fontsize=14)
    sns.despine()


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
    output_path: str = "./predictions.png",
) -> None:
    """Render test samples side by side with ground-truth and predicted masks.

    The first ``num_samples`` rows of ``test_df`` are selected, every image is
    run through ``model``, and a grid with three columns (image / image + true
    mask / image + predicted mask) is exported to a PNG file. The model's
    ``(20, H, W)`` logits are reduced to a single class id per pixel via
    ``argmax`` so they can be coloured with ``CLASS_COLORS`` and compared to
    the ground-truth color labels.

    Steps
    -----
    1. Select the first ``num_samples`` rows from the evaluation DataFrame.
    2. Pass each image through the model in inference mode to generate logits.
    3. Colour the ground-truth and predicted masks with the BDD100k palette.
    4. Export comparison subplots to a PNG file and close the figure.

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
        Destination of the exported PNG. Defaults to ``"./predictions.png"``.

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
    # Use the underlying unwrapped module for batch_size=1 inference.
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
        true_mask_color = (
            colorize_mask(class_mask, CLASS_COLORS).astype(np.float32) / 255.0
        )

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

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    # Print current Torch package versions
    print("Package versions:")
    print("*" * 26)
    print(f"torch \t\t - {torch.__version__}")
    print(f"torchvision \t - {torchvision.__version__}")

    train_df, val_df, test_df = load_dataset_from_files()

    train_transforms = A.Compose(
        [
            A.Resize(
                height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.5),
            # The mask is now a 2-D class-index map, so ``ToTensorV2`` needs no
            # ``transpose_mask``: it leaves the (H, W) mask as-is and only converts
            # the image to (C, H, W).
            ToTensorV2(),
        ]
    )

    inference_transforms = A.Compose(
        [
            A.Resize(
                height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH
            ),
            ToTensorV2(),
        ]
    )
    train_ds = BDDSegmentationDataset(train_df, transform=train_transforms)
    val_ds = BDDSegmentationDataset(val_df, transform=inference_transforms)

    # Class weights are always precomputed by ``precompute_weights.py`` and
    # shipped with the dataset, so load them directly instead of recomputing.
    print(f"Reusing precomputed sample weights from '{Path.SAMPLE_WEIGHTS_PATH}'...")
    train_sample_weights = np.load(Path.SAMPLE_WEIGHTS_PATH)

    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights.tolist(),
        num_samples=len(train_ds),
        replacement=True,
    )

    # ``shuffle`` and ``sampler`` are mutually exclusive in ``DataLoader``; the
    # sampler already provides the weighted random ordering.
    # ``num_workers`` loads batches asynchronously across background processes,
    # and ``pin_memory`` accelerates host-to-device transfers over PCIe.
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=Configuration.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Configuration.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=Configuration.BATCH_SIZE,
        shuffle=Configuration.APPLY_SHUFFLE,
        num_workers=Configuration.NUM_WORKERS,
        pin_memory=True,
    )

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=Configuration.CHANNELS,
        classes=Configuration.NUM_CLASSES,
    )
    model = model.to(Configuration.DEVICE)

    print(
        summary(
            model=model,
            input_size=(
                Configuration.BATCH_SIZE,
                Configuration.CHANNELS,
                Configuration.IMAGE_HEIGHT,
                Configuration.IMAGE_WIDTH,
            ),
            col_names=["output_size", "num_params", "trainable"],
            col_width=30,
            row_settings=["var_names"],
            depth=5,
        )
    )

    # Wrap model in DataParallel if 2 or more GPUs are present
    if torch.cuda.device_count() > 1:
        print(f"Utilizing {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)

    # Define the natural loss. An :class:`CompoundLoss` ``nn.Module`` is used
    # rather than a bare function so it satisfies the ``natural_loss: nn.Module``
    # annotation on :class:`TradesLoss`. It also collapses the one-hot masks it
    # receives from the TRADES loss back into class indices automatically.
    natural_loss = CompoundLoss(
        dice_weight=0.5,
        focal_weight=1.0,
        ignore_index=19,
    )

    loss_fn = TradesLoss(natural_loss=natural_loss)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Configuration.LR,
    )

    # Define Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        patience=Configuration.PATIENCE,
    )

    print("Training U-Net Model")
    print(f"Train on {len(train_df)} samples, validate on {len(val_df)} samples.")
    print("----------------------------------")

    # Generate training session config
    session_config = {
        "model": model,
        "train_dataloader": train_loader,
        "eval_dataloader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "eval_loss_fn": loss_fn.natural_loss,
        "epochs": Configuration.EPOCHS,
        "train_device": Configuration.DEVICE,
        "eval_device": Configuration.DEVICE,
    }

    # Execute Training Session
    model, unet_session_history = train(**session_config)

    # Create Model directory
    model_name = "teacher"
    model_path = "./model/"
    os.makedirs(model_path, exist_ok=True)

    # Save Model
    torch.save(model, os.path.join(model_path, model_name + ".pt"))

    # Convert U-Net history dict to DataFrame
    unet_session_history_df = pd.DataFrame(unet_session_history)
    print(unet_session_history_df)

    # Plot U-Net Session Training History
    plot_training_curves(
        unet_session_history,
        fig_size=(20, 20),
    )

    # Export a grid of test samples (image / image+true mask /
    # image+predicted mask) so the model output can be inspected visually.
    visualize_predictions(
        model,
        test_df,
        torch.device(Configuration.DEVICE),
    )


if __name__ == "__main__":
    main()
