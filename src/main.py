#export CUDA_LAUNCH_BLOCKING=1
import torch
import torch.nn as nn
import torchvision
from utils import image_transforms_operations
from utils import dataloader
from models import CNNModel, CustomModelWithTransformerEncoder
from utils.trainProcess import trainning_and_val_process


## Parâmetros iniciais

# Nº de classes
num_classes=10
# Quantidade de amostras por batch
batch_size=2
# Quantidade de threads dedicadas ao processamento dos dados inicialmente
num_workers=8
# learning rate
lr=0.01
# Momentum - Parâmetro do treinamento
momentum=0.9
# Número de épocas do treinamento
num_epochs=5
# Dimensões das imagens
heigth=28
width=28
channels=1

# Datasets
train_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=True, transform=image_transforms_operations.image_transforms())
val_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=False, transform=image_transforms_operations.image_transforms())

# Preprocessing
train_dataset_loader = dataloader.dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_dataset_loader = dataloader.dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Importação do modelo
# net = CNNModel.ClassifierModel(num_classes)
net = CustomModelWithTransformerEncoder.CustomImageClassifier(num_layers_attention=2, heigth=heigth, width=width, channels=channels, batch_size=batch_size)
# Caso haja GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Processo de treinamento do modelo
trainning_and_val_process(train_dataset_loader, val_dataset_loader, net, device, lr=lr, momentum=momentum, num_epochs=num_epochs)