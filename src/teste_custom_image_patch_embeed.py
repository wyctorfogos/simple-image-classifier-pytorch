import torch
import torch.nn as nn
from models import PatchEmbeed

# Parâmetros
batch_size = 4
channels = 3  # Imagens RGB
height = 28
width = 28
patch_size = 14
embed_dim = 768

# Dados fictícios
images = torch.randn(batch_size, channels, height, width)

# Instância do Patcher
patcher = PatchEmbeed.ImagePatcher(patch_size=patch_size, embed_dim=embed_dim)

# Realizar o patching
patches = patcher(images)
print("Shape dos patches:", patches.shape)  # Esperado: [batch_size, num_patches, embed_dim]
