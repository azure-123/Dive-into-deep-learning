import torch
from torch.distributions import multinomial
from d2l import torch as d2l

# 模拟投骰子的过程，对其进行抽样
fair_probs = torch.ones([6]) / 6 # 六个面概率相等
print(multinomial.Multinomial(1,fair_probs).sample()) # 以相等的概率进行一次抽样
print(multinomial.Multinomial(10,fair_probs).sample()) # 以相等的概率进行十次抽样
counts = multinomial.Multinomial(1000,fair_probs).sample()
print(counts) # 以相等的概率进行1000次抽样
print(counts / 1000) # 求出每个面的概率'

# 进行500次实验，每次实验10次抽样
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
print(cum_counts) # 每次实验后的累计次数
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(estimates) # 每次实验后的概率分布

# 作图，实验次数越多，概率越接近1/6
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()