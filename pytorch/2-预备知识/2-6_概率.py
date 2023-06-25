import torch
from torch.distributions import multinomial
from d2l import torch as d2l

# 模拟投骰子的过程，对其进行抽样
fair_porbs = torch.ones([6]) / 6 # 六个面概率相等
print(multinomial.Multinomial(1,fair_porbs).sample()) # 以相等的概率进行一次抽样
print(multinomial.Multinomial(10,fair_porbs).sample()) # 以相等的概率进行十次抽样
counts = multinomial.Multinomial(1000,fair_porbs).sample()
print(counts) # 以相等的概率进行1000次抽样
print(counts / 1000) # 求出每个面的概率