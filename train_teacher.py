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

from jaxtyping import Float, jaxtyped
from beartype import beartype


# Shape aliases that document the tensor layout at each stage of the pipeline.
#
# ``h``/``w`` are the spatial dimensions, ``b`` is the batch size and ``c`` is
# the number of channels/classes. These aliases are enforced at runtime by
# ``jaxtyped`` + ``beartype``, so a shape mismatch raises a ``TypeCheckError``
# instead of a confusing downstream broadcast error.
ImageTensor = Float[Tensor, "3 h w"]  # single image, channel-first layout
MaskTensor = Float[Tensor, "3 h w"]  # single RGB color-label mask
BatchImage = Float[Tensor, "b 3 h w"]  # collated batch of images
BatchMask = Float[Tensor, "b 3 h w"]  # collated batch of masks
Logits = Float[Tensor, "b c h w"]  # model output, ``c == NUM_CLASSES``
ClassMask = Float[Tensor, "c h w"]  # per-sample mask/prediction, ``c`` channels
Scalar = Float[Tensor, ""]  # scalar (0-dim) tensor
NumpyImage = Float[np.ndarray, "h w 3"]  # single image/mask, channels-last layout


class Configuration:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_DEVICES = 1
    NUM_WORKERS= 2

    NUM_CLASSES = 20
    EPOCHS = 20
    BATCH_SIZE = (
        32 if torch.cuda.device_count() < 2
        else (32 * torch.cuda.device_count())
    )
    LR = 1e-4
    PATIENCE = 8

    APPLY_SHUFFLE=True
    SEED = 768
    HEIGHT = 1280
    WIDTH = 720
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
    def load_sample(self, index: int) -> tuple[NumpyImage, NumpyImage]:
        """Load and normalise the image/mask pair at position ``index``.

        Both files are opened as RGB before being converted to NumPy arrays.
        Forcing RGB ensures that sources carrying an alpha channel (some
        BDD100k color-label PNGs are RGBA) are collapsed to 3 channels, which
        keeps every sample the same shape and prevents the DataLoader from
        failing to collate a batch.

        Parameters
        ----------
        index : int
            Zero-based position of the sample to load.

        Returns
        -------
        tuple[NumpyImage, NumpyImage]
            The ``(image, mask)`` pair as float32 arrays scaled to ``[0, 1]``.
            Both arrays have shape ``(H, W, 3)`` after the RGB conversion.

        Raises
        ------
        IndexError
            If ``index`` is out of the range of the dataset lists.
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Force RGB so that RGBA images (e.g. some BDD100k color-label
        # PNGs carry an alpha channel) are reduced to 3 channels. Without
        # this, mixed RGB/RGBA sources produce inconsistent channel counts
        # that later break batch collation in the DataLoader.
        image_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("RGB")

        image = np.array(image_pil).astype(np.float32) / 255.0
        mask = np.array(mask_pil).astype(np.float32) / 255.0

        return image, mask


    def __len__(self) -> int:
        return len(self.image_paths)


    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> tuple[ImageTensor, MaskTensor]:
        """Return the transformed ``(image, mask)`` pair at position ``index``.

        ``ToTensorV2`` already converts each array from ``(H, W, C)`` to the
        channel-first layout ``(C, H, W)`` that PyTorch expects. The DataLoader
        adds the batch dimension when collating samples, so no extra
        ``unsqueeze`` is applied here; doing so would yield a 5-D mask that no
        longer matches the 4-D model output.

        Parameters
        ----------
        index : int
            Zero-based position of the sample to load.

        Returns
        -------
        tuple[ImageTensor, MaskTensor]
            The ``(image, mask)`` pair as tensors of shape ``(3, H, W)``.
        """
        image, mask = self.load_sample(index)

        # Transform if necessary
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
        else:
            transformed = ToTensorV2(image=image, mask=mask)
        return transformed["image"], transformed["mask"]


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


def execute_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device) -> tuple[float, float]:

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
        predicted_class = torch.sigmoid(y_pred)
        predicted_class = (predicted_class > 0.5).float()

        eps = 1e-8
        train_dice += (
            (2 * (y * predicted_class).sum() + eps) /
            ((y + predicted_class).sum() + eps)
        ).cpu().item()


    # Compute Step Metrics
    train_loss = train_loss / len(dataloader)
    train_dice = train_dice / len(dataloader)

    return train_loss, train_dice


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    loss_fn: nn.Module,
    device: torch.device) -> tuple[float, float]:

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
            predicted_class = torch.sigmoid(y_pred)
            predicted_class = (predicted_class > 0.5).float()

            eps = 1e-8
            eval_dice += (
                (2 * (y * predicted_class).sum() + eps) /
                ((y + predicted_class).sum() + eps)
            ).cpu().item()

    # Compute Step Metrics
    eval_loss = eval_loss / len(dataloader)
    eval_dice = eval_dice / len(dataloader)

    return eval_loss, eval_dice


def train(
    model: nn.Module,
    train_dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    eval_dataloader: DataLoader[tuple[ImageTensor, MaskTensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler.ReduceLROnPlateau | None,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device) -> Dict[str, List[float]]:

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
            predicted_class = torch.sigmoid(y_pred)
            predicted_class = (predicted_class > 0.3).float()

            # Compute Batch Metrics For Each Mask
            for true_mask, pred_mask in zip(y, predicted_class, strict=True):
                iou = jaccard_index(true_mask, pred_mask).cpu().item()
                dice = dice_score(true_mask, pred_mask).cpu().item()

                # Record metrics
                metrics['dice_score'].append(dice)
                metrics['IoU'].append(iou)

    return metrics


def main() -> None:
    # Print current Torch package versions
    print('Package versions:')
    print('*'*26)
    print(f'torch \t\t - {torch.__version__}')
    print(f'torchvision \t - {torchvision.__version__}')

    train_df, val_df, test_df = load_dataset_from_files()

    train_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
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
                input_size=(Configuration.BATCH_SIZE, Configuration.CHANNELS, Configuration.WIDTH, Configuration.HEIGHT),
                col_names=["output_size", "num_params", "trainable"],
                col_width=30,
                row_settings=["var_names"],
                depth=5
            )
    )

    # Define Loss Function
    loss_fn = nn.CrossEntropyLoss()

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


if __name__ == "__main__":
    main()
