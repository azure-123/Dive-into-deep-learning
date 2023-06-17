import torch

# 四则运算
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x**y)

# 用一维张量表示向量
x = torch.arange(4)
print(x)

# 通过索引访问张量元素
print(x[3])

# 访问张量的长度
print(len(x))

# 访问张量的形状
print(x.shape)

# 实例化矩阵
A = torch.arange(20).reshape(5, 4)
print(A)

# 访问矩阵的转置
print(A.T)

# 对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)

# 对称矩阵和其转置作比较
print(B == B.T)

# 三维张量
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 两个相同形状的张量相加
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A, A + B)

# 矩阵的Hadamard积（按元素相乘）
print(A * B)

# 张量与标量相加与相乘
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

# 向量元素求和
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())

# 张量元素求和
print(A.shape, A.sum())

# 沿0轴降维求和，轴0在输出的形状中消失
A_sum_axis0 = A.sum(dim = 0)
print(A_sum_axis0, A_sum_axis0.shape)

# 沿1轴降维求和，轴1在输出的形状中消失
A_sum_axis1 = A.sum(dim = 1)
print(A_sum_axis1, A_sum_axis1.shape)

# 沿着行和列对矩阵求和
print(A.sum(dim = [0, 1]))

# 求矩阵的平均值
print(A.mean(), A.sum() / A.numel())

# 沿着行和列对矩阵求平均值
print(A.mean(dim = 0), A.sum(dim = 0) / A.shape[0])
print(A.mean(dim = 1), A.sum(dim = 1) / A.shape[1])

# 非降维求和
sum_A = A.sum(dim = 1, keepdim=True)
print(sum_A)

# 通过广播相除
print(A / sum_A)

# 沿0轴求累积和
print(A.cumsum(dim=0))

# 求向量点积
y = torch.ones(4, dtype = torch.float32)
print(torch.dot(x, y)) # 使用点乘函数
print(torch.sum(x * y)) # 使用向量元素相乘

# 矩阵和向量相乘
print(A.shape, x.shape, torch.mv(A, x))

# 矩阵和矩阵相乘
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 计算向量的L2范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# 计算向量的L1范数
print(torch.abs(u).sum())

# 计算矩阵的Frobenius范数（相当于矩阵的L2范数）
print(torch.norm(torch.ones(4,  9)))