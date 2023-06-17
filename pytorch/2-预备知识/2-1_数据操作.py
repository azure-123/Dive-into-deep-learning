import torch

# 创建0开始的行向量
x = torch.arange(12)
print(x)

# 获得x的形状
print(x.shape)

# 获取x的元素个数
print(x.numel())

# 改变x的形状
X = x.reshape(3,4)
print(X)

# 创建一个全0张量
print(torch.zeros((2,3,4)))

# 创建一个全1张量
print(torch.ones((2,3,4)))

# 创建一个均值为0、标准差为1的高斯分布
print(torch.randn(3,4))

# 赋予张量特定值
print(torch.tensor([[1,2,3,4],[4,3,2,1],[2,1,4,3]]))

# 四则运算和求幂运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

# 求e的幂
print(torch.exp(x))

# 张量的连结
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X,Y),dim=0)) # 沿行连结（增加行数）
print(torch.cat((X,Y),dim=1)) # 沿列连结（增加列数）

# 判断两个张量相等的位置
print(X==Y)

# 对张量所有元素求和
print(X.sum())

# 广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a,b)
print(a + b)

# 张量的切片
print(X[-1], X[1:3])

# 利用索引对张量赋值
X[1, 2] = 9
print(X)

# 为多个元素赋相同的值
X[0:2, :] = 12
print(X)

# 赋值时分配新的内存
before = id(Y)
Y = Y + X
print(id(Y) == before)

# 节省内存（使用切片）
before = id(Y)
Y[:] = Y + X
print(id(Y) == before)

# 节省内存（使用原地运算）
before = id(Y)
Y += X
print(id(Y) == before)

# 与numpy.ndarray的互相切换
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

# 大小为1的张量转换为python标量
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))

