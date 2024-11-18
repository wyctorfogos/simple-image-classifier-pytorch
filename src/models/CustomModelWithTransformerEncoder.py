import torch
import torch.nn as nn
from models import TransformerModel
from models import PatchEmbeed

class CustomImageClassifier(nn.Module):
    def __init__(self, embed_dim=784, num_heads=8, num_layers_attention=2, mlp_dim=2048, patch_size=14, num_classes=10, dropout=0.2, batch_size=4, channels=3, heigth=224, width=224):
        super().__init__()

        self.patch_embeed_image = PatchEmbeed.ImagePatcher(embed_dim=embed_dim, batch_size= batch_size, patch_size=patch_size, channels=channels, height=heigth, width=width)
        # Bloco Transformer com múltiplas camadas
        self.transformer_block = TransformerModel.TransformerEncoderBlock(
            img_dim=embed_dim, 
            num_layers=num_layers_attention,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        
        # Cabeçalho MLP para classificação
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),  # Normalização
            nn.Linear(embed_dim, num_classes)  # Projeção para o número de classes
        )

    def forward(self, x):
        # Uso do patch das imagens antes de extrair as features
        x = self.patch_embeed_image(x)
        # Entrada já no formato [batch_size, seq_len, embed_dim]
        x = self.transformer_block(x)  # Saída [batch_size, seq_len, embed_dim]

        # Agregação: Média de todos os tokens
        x = x.mean(dim=1)  # [batch_size, embed_dim]

        return self.mlp_head(x)
