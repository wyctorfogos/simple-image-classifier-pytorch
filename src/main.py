#export CUDA_LAUNCH_BLOCKING=1
import torch
import torch.nn as nn
import torchvision
from utils import image_transforms_operations
from utils import dataloader
from models import CNNModel, CustomModelWithTransformerEncoder, VGG16Model, RESNET18Model
from utils.trainProcess import trainning_and_val_process


## Parâmetros iniciais

# Nº de classes
num_classes=10
# Quantidade de amostras por batch
batch_size=32
# Quantidade de threads dedicadas ao processamento dos dados inicialmente
num_workers=8
# learning rate
lr=1e-4
# Momentum - Parâmetro do treinamento
momentum=0.9
# Número de épocas do treinamento
num_epochs=5
# Dimensões das imagens
heigth=28
width=28
channels=3
# Caso haja GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
train_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=True, transform=image_transforms_operations.image_transforms())
val_dataset = torchvision.datasets.FashionMNIST('../dataset', download=True, train=False, transform=image_transforms_operations.image_transforms())

# Preprocessing
train_dataset_loader = dataloader.dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_dataset_loader = dataloader.dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Importação do modelo
# net = CNNModel.ClassifierModel(num_classes) ## Modelo convolucional simples
# net = CustomModelWithTransformerEncoder.CustomImageClassifier(num_layers_attention=8, mlp_dim=1024, heigth=heigth, width=width, channels=channels, batch_size=batch_size, device=device) ## Modelo com um encoder do transformers
# net = VGG16Model.VGG16Model(num_classes=num_classes, device=device)

net = RESNET18Model.RESNET18Model(num_classes=num_classes, device=device) # Uso do modelo Resnet18

# Carregue o modelo no cpu ou na gpu, caso tenha sido detectada
net.to(device)

# Processo de treinamento do modelo
trainning_and_val_process(train_dataset_loader, val_dataset_loader, net, device, lr=lr, momentum=momentum, num_epochs=num_epochs)