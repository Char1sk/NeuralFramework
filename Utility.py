import numpy as np


# 进行字符串到函数的映射
def strToFunc(funcname):
    if funcname == 'linear':
        return (linear, dLinear)
    if funcname == 'sigmoid':
        return (sigmoid, dSigmoid)
    if funcname == 'meanSquareError':
        return (meanSquareError, dMeanSquareError)
    if funcname == 'crossEntropy':
        return (softmaxCrossEntropy, dSoftmaxCrossEntropy)


# 【激活函数】硬极限函数
def hardlim(x):
    x = np.where(x > 0, 1, 0)
    return x


# 【激活函数】对称硬极限函数
def hardlims(x):
    x = np.where(x > 0, 1, -1)
    return x


# 【激活函数】线性函数
def linear(x):
    return x


# 【激活函数导数】线性函数
def dLinear(x):
    return np.ones_like(x)


# 【激活函数】Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 【激活函数导数】Sigmoid函数
def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 【损失函数】均方误差函数
def meanSquareError(a, y):
    return 1 / 2 * np.sum((a - y) ** 2)


# 【损失函数导数】均方误差函数
def dMeanSquareError(a, y):
    return a - y


# 【损失函数】交叉熵函数(最后一层Softmax激活)
def softmaxCrossEntropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


# 【损失函数导数】交叉熵函数(最后一层Softmax激活)
def dSoftmaxCrossEntropy(a, y):
    return a - y
