import torch
from torch import nn

# 网络结构
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 查看某一层的权重
print(net[2].state_dict())
# 查看某一层的偏置
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)