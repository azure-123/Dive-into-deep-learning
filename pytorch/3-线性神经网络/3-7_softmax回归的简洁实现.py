import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 创建模型，在线性层之前定义展平层，调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

# 定义损失函数（交叉熵函数）
loss = nn.CrossEntropyLoss(reduce='mean')

# 算法优化器
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

#开始训练
num_epochs = 10
d2l.train_ch3(net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=trainer)
d2l.plt.show()

d2l.predict_ch3(net, test_iter)
d2l.plt.show()