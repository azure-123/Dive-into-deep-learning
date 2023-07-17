import torch
from d2l import torch as d2l
from torch import nn

num_inputs, num_outputs, num_hidden = 28*28, 10, 256
batch_size = 256

# 获取数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 定义网络结构
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_inputs, num_hidden),
                    nn.ReLU(),
                    nn.Linear(num_hidden, num_inputs))

# 初始化参数
def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)

# 定义损失函数
loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
# 算法优化器
updater = torch.optim.SGD(net.parameters(), lr=lr)

# 开始训练
d2l.train_ch3(net=net,train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=updater)

# 开始预测
d2l.predict_ch3(net=net, test_iter=test_iter)
d2l.plt.show()