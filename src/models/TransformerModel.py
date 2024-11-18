import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, img_dim, num_layers=2):
        super(TransformerEncoderBlock, self).__init__()
        self.num_layers=num_layers
        self.img_dim=img_dim
        self.nhead=8
        self.dropout=0.1
        self.dim_feedforward=512
        self.activation="relu"
        self.layer_norm_eps=1e-4
        self.batch_first=False
        self.norm_first=False
        self.bias=True
        self.encoderLayers=self.encoderLayersConfig()
        self.transformEncoder=nn.TransformerEncoder(self.encoderLayers, num_layers=self.num_layers)
        
    def forward(self, x):
        x = self.transformEncoder(x)
        return x
    
    def encoderLayersConfig(self):
        return nn.TransformerEncoderLayer(d_model=self.img_dim, nhead=8, dropout=self.dropout, activation=self.activation, dim_feedforward=self.dim_feedforward, layer_norm_eps=self.layer_norm_eps, batch_first=self.batch_first, norm_first=self.norm_first )