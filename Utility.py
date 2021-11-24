import numpy as np


# 【激活函数】硬极限函数
def hardlim(x):
    x = np.where(x > 0, 1, 0)
    return x
# 【激活函数】硬极限函数 这个有三段 >0 1 =0 0 <0 -1
def hardlims(x):
    x = np.where(x > 0, 1, np.where(x == 0, 0, -1))
    return x

# 【激活函数】一些其他的激活函数
def RENAME_THIS_ACTIVATION_FUNCTION(x):
    pass


# 【损失函数】均方误差函数
def meanSquareError(a, y):
    m = a.shape[1]
    return np.sum(np.square(y-a))/m/2


# 【损失函数】一些其他的损失函数
def RENAME_THIS_COST_FUNCTION(x):
    pass
