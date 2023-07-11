import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 创建模型
net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

# 初始化权重
def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

# 定义损失函数（交叉熵函数）
loss = nn.CrossEntropyLoss(reduce='None')

# 算法优化器
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

#开始训练
num_epochs = 10
d2l.train_epoch_ch3(net, train_iter=train_iter,loss=loss, updater=trainer)