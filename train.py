import random
import os
import glob
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
import torch.optim.lr_scheduler as lr_scheduler

from torch import nn
from torch.utils.data import (Dataset, DataLoader)
from torchvision import transforms
from torchinfo import summary
from torchview import draw_graph

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    accuracy_score, f1_score, matthews_corrcoef, 
    confusion_matrix, ConfusionMatrixDisplay
)


class Configuration:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_DEVICES = 1
    NUM_WORKERS= 2

    NUM_CLASSES = 2
    EPOCHS = 100
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

    # TODO: add adversary
    train_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])

    inference_transforms = A.Compose([
        ToTensorV2(),
    ])
    train_set = BDDSegmentationDataset(train_df, transform=train_transforms)
    test_set = BDDSegmentationDataset(test_df, transform=inference_transforms)


if __name__ == "__main__":
    main()
