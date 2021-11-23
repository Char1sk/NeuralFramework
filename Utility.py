import numpy as np


# 【激活函数】硬极限函数
def hardlim(x):
    x = np.where(x > 0, 1, 0)
    return x


# 【激活函数】一些其他的激活函数
def RENAME_THIS_ACTIVATION_FUNCTION(x):
    pass


# 【损失函数】均方误差函数
def meanSquareError(a, y):
    pass


# 【损失函数】一些其他的损失函数
def RENAME_THIS_COST_FUNCTION(x):
    pass
