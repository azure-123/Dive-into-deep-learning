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

# 将y作为y_hat中的概率索引
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])

# 定义损失函数，为交叉熵函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y]) # y_hat取其每个样本的索引，y为每个样本的正确分类，将这个正确分类套到y_hat当中，看这个分类在y_hat当中的概率为多少

print(cross_entropy(y_hat, y)) # 第一个正确分类是第0个，但是y_hat中其概率特别小，说明预测错误，返回值很大；第二个预测正确了，返回值很小

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    cmp = y_hat.type(y.dtype) == y # 将y_hat和y的元素转化为一致再比较
    return float(cmp.type(y.dtype).sum()) # 由于cmp都是True和False组成的tensor，因此转化为int类型

print(accuracy(y_hat, y) / len(y))

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 网络设置为评估模式
    metric = Accumulator(2) # 创建一个累加器，第一个元素为预测正确的数量，第二个元素为预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # 每个迭代轮数都添加预测正确的数量和总数
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, torch.nn.Module):
        net.train() # 将网络设置为训练模式
    metric = Accumulator(3) # 创建一个累加器，第一个元素为总损失，第二个元素为预测正确的数量，第三个元素为预测总数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer): # 若updater是torch模块中自带的
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else: # 若为自己定义的updater
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2] # 返回平均损失和准确度

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, accuracy {test_acc:f}') # 打印该轮的损失

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr=lr, batch_size=batch_size)

num_epochs = 10
train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, loss=cross_entropy, num_epochs=num_epochs,updater=updater)

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net=net,test_iter=test_iter)


