import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 沿用3.2的数值和函数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 使用DataLoader加载样本
def load_array(data_array, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array([features, labels], batch_size=batch_size)
print(next(iter(data_iter)))

# 导入nn包
from torch import nn

# 定义预测模型
net = nn.Sequential(nn.Linear(2,1)) # Sequential表示网络的一层，linear表示预测模型为线性，2为输入特征维度，1为输出维度

# 初始化参数
net[0].weight.data = torch.randn(1, 2)
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data = torch.zeros(1)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 实例化一个SGD实例
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epoch = 3
# 开始训练
for epoch in range(num_epoch):
    for X, y in data_iter:
        l = loss(net(X), y) # 计算一小批的损失函数
        trainer.zero_grad() # 让梯度清零
        l.backward() # 对损失函数进行反向传播求梯度
        trainer.step() # 进行算法优化
    train_l = loss(net(features), labels) # 计算所有样本的损失函数来衡量参数
    print(f'epoch {epoch + 1}, loss {train_l:f}') # 打印该轮的损失