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
from controllers.trainProcess import trainning_process


# Datasets
train_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=True, transform=image_transforms())
val_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=False, transform=image_transforms())

# Preprocessing
train_dataset_loader= dataloader.dataloader(train_dataset, shuffle=True)
val_dataset_loader= dataloader.dataloader(val_dataset, shuffle=False)

# Importação do modelo
net = CNNModel.ClassifierModel()

# Caso haja GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Processo de treinamento do modelo
trainning_process(train_dataset_loader, net, device)