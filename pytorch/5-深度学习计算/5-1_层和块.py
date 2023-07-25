import torch
from torch import nn
from torch.nn import functional as F

# MLP结构
net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
X = torch.rand(2, 20)
print(net(X))