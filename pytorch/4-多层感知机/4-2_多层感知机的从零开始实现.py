import torch
from d2l import torch as d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 各个层涉及的维度
num_inputs, num_outputs, num_hidden = 28*28, 10, 256

# 网络参数
W1 = nn.Parameter(torch.randn(num_inputs, num_hidden, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hidden, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 将网络参数放入list中
params = [W1, b1, W2, b2]

# 定义relu函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 定义网络模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)

# 定义损失函数，为交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 定义算法优化器
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params=params, lr=lr)

# 开始训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 进行预测
d2l.predict_ch3(net, test_iter)
d2l.plt.show()