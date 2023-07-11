import torch
from IPython import display
from d2l import torch as d2l

# 加载训练和测试数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 初始化权重和偏置参数
num_inputs = 28 * 28
num_outputs = 10
W = torch.normal(0, 0.01, (num_inputs, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)

# 定义softmax函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return torch.exp(X) / partition

# 检测softmax函数的正确性
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(dim=1))

# 定义网络，需要将图像拉长为28*28
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

