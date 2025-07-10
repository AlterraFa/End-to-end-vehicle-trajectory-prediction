import torch
from torch import nn
from utils.CBAM import ChannelPool

class LinChannelGate(torch.nn.Module):  
    def __init__(self, channelDim, reductionRatio = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channelDim, channelDim // reductionRatio),
            nn.ReLU(),
            nn.Linear(channelDim // reductionRatio, channelDim)
        )
        
    def forward(self, x):
        score = self.mlp(x)
        score *= 2
        return x * torch.sigmoid(score)
    
class LinSpatialGate(torch.nn.Module):
    def __init__(self):
        super(LinSpatialGate, self).__init__()
        self.pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size = 1),
            nn.BatchNorm2d(1, eps = 1e-5, momentum = 0.01, affine = True),
            nn.ReLU(),
        )
        
    def forward(self, x):
        score = self.pool(x)
        score = self.conv(score[:, :, None, None]).squeeze()[:, None]
        return x * torch.sigmoid(score)
        
        
class LinCBAM(torch.nn.Module):
    def __init__(self, channelDim, reductionRatio = 16):
        super(LinCBAM, self).__init__()
        self.channelAtten = LinChannelGate(channelDim, reductionRatio)
        self.spatialAtten = LinSpatialGate()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        score = self.channelAtten(x)
        score = self.spatialAtten(score)
        x = self.dropout(x + score)
        return x