import math
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

max_degree = 20 # 多项式最大阶数
n_train, n_test =100, 100 # 训练和测试数据量
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1)) # 多项式中的自变量
np.random.shuffle(features)

# 计算多项式的因变量
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)

labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape) #添加一定误差

# 将以上numpy格式数据转化为tensor
labels, features, poly_features, true_w = [torch.tensor(x, dtype=torch.float32) for x in [labels, features, poly_features, true_w]]

print(features[:2], poly_features[:2, :], labels[:2])

# 评估模型的损失
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# 训练过程
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) # 不使用偏置，因为已经在多项式中设置了
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape((-1,1))), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape((-1,1))), batch_size, is_train=False)
    updater = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
        print('epoch:', epoch + 1)
    print('weight:', net[0].weight.data.numpy())

# 选择前四个维度训练，是恰当的维度
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
d2l.plt.show()

# 选择前两个维度训练，欠拟合
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

# 从多项式特征中选取所有维度，过拟合
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)