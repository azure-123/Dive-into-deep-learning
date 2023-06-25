import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l

# 定义一个函数
def f(x):
    return 3 * x * x - 4 * x

# 求极限
def numeric_limit(f, x, h):
    return (f(x + h) - f(x)) / h

# 让h越来越小，看在x=1极限越接近多少
h = 0.1
for i in range(5):
    print(f'h = {h: .5f}, numeric_limit = {numeric_limit(f, 1, h): .5f}')
    h *= 0.1

