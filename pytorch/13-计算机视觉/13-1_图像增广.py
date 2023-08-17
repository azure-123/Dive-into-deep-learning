import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 展示图片
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)

# 查看数据增强的效果
def apply(img, aug, num_rows = 2, num_cols = 4, scale = 1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale)

# 翻转和裁剪
# 水平翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
# 垂直翻转
apply(img, torchvision.transforms.RandomVerticalFlip())