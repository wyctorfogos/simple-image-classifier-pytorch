import torch
import torch.nn as nn
from torchvision import models

class VGG16Model(nn.Module):
    def __init__(self, num_classes, device):
        super(VGG16Model, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.pretrained = True
        self.model = self.load_model()
    
    def load_model(self):
        # Primeiro o modelo VGG16 pre-trainano é carregado.
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=self.pretrained)
        
        # Remover a última camada de pooling
        model.features = nn.Sequential(*list(model.features.children())[:-1])

        # Ajustar a camada inicial para aceitar 1 canal em vez de 3
        model.features[0] = nn.Conv2d(
            in_channels=1,  # Grayscale (1 canal)
            out_channels=64,  # Mantém os canais de saída originais
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Congelamento dos pesos
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Tirar o último layer pelo desejado
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=self.num_classes)
        
        # Adiciona o modelo ao device em uso
        model = model.to(self.device)
        return model
    
    def forward(self, x):
        return self.model(x)
