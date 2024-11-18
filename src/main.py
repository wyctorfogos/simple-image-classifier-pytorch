#export CUDA_LAUNCH_BLOCKING=1

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from controllers.transforms import image_transforms
from models import dataloader
from models import CNNModel
from controllers.trainProcess import trainning_and_val_process


# Datasets
train_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=True, transform=image_transforms())
val_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=False, transform=image_transforms())

# Preprocessing
train_dataset_loader= dataloader.dataloader(train_dataset, num_workers=6, shuffle=True)
val_dataset_loader= dataloader.dataloader(val_dataset, num_workers=6, shuffle=False)

# Nº de classes
num_classes=10

# Importação do modelo
net = CNNModel.ClassifierModel(num_classes)

# Caso haja GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Processo de treinamento do modelo
trainning_and_val_process(train_dataset_loader, val_dataset_loader, net, device, lr=0.01, momentum=0.9, num_epochs=5)