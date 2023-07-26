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

# 指定常数参数
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        if X.abs().sum() > 1:
            X /= 2
        return X.sum()
net = FixedHiddenMLP()
print(net(X))

# 混合搭配组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))