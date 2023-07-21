import torch
from torch import nn
from d2l import torch as d2l

# 定义dropout层
def dropout_layer(X, dropout):
    assert dropout>=0 and dropout<=1
    # dropout为1，说明完全丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # dropout为0，说明全不丢弃
    if dropout == 0:
        return X
    mask = (torch.randn(X.shape) > dropout).float()
    return mask * X / (1 - dropout)

# 测试dropout层
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

# 设置参数
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 28 * 28, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

# 从零开始实现
# 定义网络模型
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.is_training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.is_training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.is_training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 设置训练参数
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
updater = torch.optim.SGD(net.parameters(), lr)
# 开始训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 简洁实现
# 定义网络模型
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(num_hiddens2, num_outputs)
)

# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr)

# 开始训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.plt.show()

