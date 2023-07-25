import torch
from torch import nn
from torch.nn import functional as F

# MLP结构
net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
X = torch.rand(2, 20)
print(net(X))

# MLP模块
class MLP(nn.Module):
    def __init__(self):
        super().__init__() # 继承父类nn.Module
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
net = MLP()
print(net(X))

# 顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))