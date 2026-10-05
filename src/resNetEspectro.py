import torch.nn as nn
import torch

class ResNetEspectro(nn.Module):
    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo
        # Media y desviación de ImageNet, con forma (1, 3, 1, 1) para operar por canal
        self.register_buffer("media", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",   torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)            # (N, 1, 128, 128) → (N, 3, 128, 128)
        x = (x - self.media) / self.std      # misma normalización que en ImageNet
        return self.modelo(x)