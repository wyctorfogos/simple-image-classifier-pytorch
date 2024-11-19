import torch.nn as nn

class ImagePatcher(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, batch_size=8, channels=3, height=224, width=224, device="cpu"):
        super(ImagePatcher, self).__init__()
        self.device=device
        self.batch_size = batch_size
        self.channels = channels  # Imagens RGB
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, images):
        """
        Realiza o patching nas imagens.
        Args:
            images (torch.Tensor): Tensor de imagens de entrada com shape [batch_size, channels, height, width].
        Returns:
            torch.Tensor: Patches linearizados e projetados, com shape [batch_size, num_patches, embed_dim].
        """
        batch_size, channels, height, width = images.size()
        
        # Certifique-se de que as dimensões da imagem sejam divisíveis pelo patch_size
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            "Altura e largura da imagem devem ser divisíveis pelo tamanho do patch."

        # Divida a imagem em patches
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Reorganize as imagens em patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, num_patches, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, num_patches, channels, patch_size, patch_size]
        patches = patches.view(batch_size, num_patches, -1)  # [batch_size, num_patches, channels * patch_size^2]
        patches = patches.to(self.device)
        # Projete os patches para a dimensão embed_dim
        linear_proj = nn.Linear(channels * (self.patch_size ** 2), self.embed_dim)
        linear_proj=linear_proj.to(self.device) 
        return linear_proj(patches)
