import torch
import torch.nn as nn
from torchvision import models

class RESNET18Model(nn.Module):
    def __init__(self, num_classes, device):
        super(RESNET18Model, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.pre_trained = True
        self.model = self.load_model()
    
    def load_model(self):
        # Carregar o modelo ResNet18 pré-treinado
        model = models.resnet18(pretrained=self.pre_trained)
        
        # Ajustar a camada inicial (conv1) para aceitar 1 canal em vez de 3
        model.conv1 = nn.Conv2d(
            in_channels=1,  # Grayscale (1 canal)
            out_channels=64,  # Mantém os canais de saída originais
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Congelar os pesos das camadas convolucionais
        for param in model.parameters():
            param.requires_grad = False
        
        # Substituir a última camada fully connected (fc) pelo número de classes desejado
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        # Mover o modelo para o dispositivo
        model = model.to(self.device)
        return model
    
    def forward(self, x):
        return self.model(x)
