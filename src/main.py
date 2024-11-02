import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from controllers.transforms import image_transforms
from models import dataloader

# Datasets
train_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=True, transform=image_transforms())
val_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=False, transform=image_transforms())

# Preprocessing
train_dataset_loader= dataloader.dataloader(train_dataset)
val_dataset_loader= dataloader.dataloader(val_dataset)


