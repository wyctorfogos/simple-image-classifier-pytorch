import torch
import models
from models import TransformerModel

# Parâmetros
img_dim = 768
num_layers = 1
seq_len = 4
batch_size = 1

# Instância do bloco Transformer Encoder
block = TransformerModel.TransformerEncoderBlock(img_dim=img_dim, num_layers=num_layers)

# Dados fictícios
dummy_input = torch.randn(seq_len, batch_size, img_dim)  # [seq_len, batch_size, img_dim]

# Teste
output = block(dummy_input)
print(output)  # Saída esperada: [seq_len, batch_size, img_dim]
