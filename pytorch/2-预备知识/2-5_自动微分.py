import torch

x = torch.arange(4.0)

# 设置x需要梯度
x.requires_grad_(True)
print(x.grad)

# y是关于x的函数
y = 2 * torch.dot(x, x)
y.backward()
print(y)
print(x.grad)

# 验证是否和4x相等
print(x.grad == 4*x)

# 计算另一个函数
x.grad.zero_()
y = x.sum()
y.backward()
print(y)
print(x.grad)

# 非标量变量的反向传播
x.grad.zero_()
y = x * x
print(y)
y.sum().backward()
print(x.grad)

# 分离计算，将u视为常数，而不作为函数中的变量计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x # 此时z=[0, x, 4x, 9x]，而不是x^3
z.sum().backward()
print(x.grad)
print(x.grad == u)

# 在Python控制流当中的自动微分
def f(a):
    b = 2 * a
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# 验证f(x)=kx，微分的结果取决于a
print(a.grad == d / a)