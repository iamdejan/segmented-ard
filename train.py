import random
import os
import glob
import time
import warnings

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
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3 # RGB

    PATH = "./data"


def main() -> None:
    # Print current Torch package versions
    print('Package versions:')
    print('*'*26)
    print(f'torch \t\t - {torch.__version__}')
    print(f'torchvision \t - {torchvision.__version__}')


if __name__ == "__main__":
    main()
