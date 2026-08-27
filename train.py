import glob

import pandas as pd
import numpy as np

import torch
import torchvision

from torch import nn
from torch.utils.data import (Dataset, DataLoader)
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

class Configuration:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_DEVICES = 1
    NUM_WORKERS= 2

    NUM_CLASSES = 2
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


class BDDSegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        super(BDDSegmentationDataset, self).__init__()

        self.image_paths = df["image_paths"].to_list()
        self.mask_paths = df["mask_paths"].to_list()
        self.transform = transform


    def load_sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0

        return image, mask


    def __len__(self) -> int:
        return self.image_paths.__len__()


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.load_sample(index)

        # Transform if necessary
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
        else:
            transformed = ToTensorV2(image=image, mask=mask)
        return transformed["image"], transformed["mask"].unsqueeze_(0)


def find_image_path_from_mask(complete_mask_path: str) -> str:
    file_path_split = complete_mask_path.split("/")
    mask_file_name = file_path_split[-1].split("_")[0]

    image_path = ImagePath.IMAGE_TRAIN_PATH + "/" + mask_file_name + ".jpg"
    return image_path


def execute_epoch(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    optimizer:torch.optim.Optimizer,
    loss_fn:torch.nn.Module,
    device:torch.device) -> tuple[float, float]:
    
    # Set model into training mode
    model.train()
    
    # Initialize train loss & accuracy
    train_loss, train_dice = 0, 0
    
    # Execute training loop over train dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Load data onto target device
        X, y = X.to(device), y.to(device)
        
        # Feed-forward and compute metrics
        y_pred = model(X)
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
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    device:torch.device) -> Tuple[float, float]:
    
    # Set model into eval mode
    model.eval()
    
    # Initialize eval loss & accuracy
    eval_loss, eval_dice = 0, 0
    
    # Active inferene context manager
    with torch.inference_mode():
        # Execute eval loop over dataloader
        for batch, (X, y) in enumerate(dataloader):
            # Load data onto target device
            X, y = X.to(device), y.to(device)

            # Feed-forward and compute metrics
            y_pred = model(X)
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
    model:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    eval_dataloader:torch.utils.data.DataLoader,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler,
    loss_fn:torch.nn.Module,
    epochs:int,
    device:torch.device) -> Dict[str, List]:
    
    # Initialize training session
    session = {
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


def predict(
    model:nn.Module, 
    sample_loader:torch.utils.data.DataLoader,
    device:torch.device,
    threshold:float=0.5) -> np.ndarray:
    
    # Set model into eval mode
    model.eval()
    
    predictions = []
    
    # Active inferene context manager
    with torch.inference_mode():
        # Execute eval loop over dataloader
        for batch, (X, y) in enumerate(tqdm(sample_loader)):
            # Load data onto target device
            X, y = X.to(device), y.to(device)

            # Feed-forward and compute metrics
            y_pred = model(X) 

            # Compute Batch Metrics
            predicted_class = torch.sigmoid(y_pred)
            predicted_class = (predicted_class >= threshold).float()

            # Record prediction
            predictions.append(predicted_class.cpu().numpy())
        
    return np.vstack(predictions)


def main() -> None:
    # Print current Torch package versions
    print('Package versions:')
    print('*'*26)
    print(f'torch \t\t - {torch.__version__}')
    print(f'torchvision \t - {torchvision.__version__}')

    # get masks
    mask_paths = glob.glob(f"{ImagePath.SEGMENTATION_MASK_TRAIN_PATH}/*.png")
    image_paths = list(map(find_image_path_from_mask, mask_paths))

    df = pd.DataFrame({
        "image_paths": image_paths,
        "mask_paths": mask_paths,
    })
    print(df[:5])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=Configuration.SEED)

    train_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])

    inference_transforms = A.Compose([
        ToTensorV2(),
    ])
    train_ds = BDDSegmentationDataset(train_df, transform=train_transforms)
    test_ds = BDDSegmentationDataset(test_df, transform=inference_transforms)

    train_loader = DataLoader(
        dataset=train_ds,
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
        classes=20
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


if __name__ == "__main__":
    main()
