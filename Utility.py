import numpy as np


# 【激活函数】硬极限函数
def hardlim(x):
    x = np.where(x > 0, 1, 0)
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 【激活函数】一些其他的激活函数
def RENAME_THIS_ACTIVATION_FUNCTION(x):
    pass


# 【损失函数】均方误差函数
def meanSquareError(a, y):
    return 1 / 2 * np.sum((a - y) ** 2)


# meanSqrop
def dMeanSquareError(a, y):
    return a - y


def softmaxCrossEntropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


def dSoftmaxCrossEntropy(a, y):
    return a - y


# 【损失函数】一些其他的损失函数
def RENAME_THIS_COST_FUNCTION(x):
    pass
