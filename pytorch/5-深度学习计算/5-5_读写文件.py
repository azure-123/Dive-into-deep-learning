import torch
from torch import nn
from torch.nn import functional as F

# 保存一个张量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(x2)

# 保存一个张量列表
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)

# 保存一个字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 保存一个模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print(Y)
torch.save(net.state_dict(), 'mlp.params')
clone = MLP() # 读取MLP的定义
clone.load_state_dict(torch.load('mlp.params')) # 读取state dict
print(clone.eval())
Y_clone = clone(X)
print(Y_clone == Y)