import torch
import random
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 生成一批数据样本
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

d2l.plt.scatter(features[:,1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()

# 获取小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) # 获取样本的索引
    random.shuffle(indices) # 打乱样本的索引，使得后面选取样本是随机的
    for i in range(0, num_examples, batch_size): 
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)]) # 从样本当中抽取一个批量，若剩下的不够一个批量，就把剩下的都取出
        yield features[batch_indices], labels[batch_indices] # 每次迭代返回一批样本

batch_size = 10

for X, y in data_iter(batch_size=batch_size,features=features, labels=labels):
    print(X, '\n', y) # 打印查看一批样本
    break

# 初始化w和b的值
w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 线性回归模型
def linreg(w, X, b):
    return torch.matmul(X, w) + b

# 损失函数，此处为均方误差
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 优化算法，小批量随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= param.grad * lr / batch_size
            param.grad.zero_()

lr = 0.03
epoch = 3
net = linreg
loss = squared_loss

# 开始训练
for i in range(epoch):
    # 每次取出一批
    for X, y in data_iter(batch_size=batch_size, features=features, labels=labels):
        l = loss(net(w, X, b), y) # 计算这一批的损失函数
        l.sum().backward() # 用损失函数求梯度
        sgd([w, b], lr=lr, batch_size=batch_size) # 使用梯度对参数进行更新
    # 使用整体的样本对参数进行评估
    with torch.no_grad():
        train_l = loss(net(w, features, b), labels) # 计算整体的损失函数
        print(f'epoch {i + 1}, loss {float(train_l.mean()):f}') # 打印该轮的均方误差损失函数
