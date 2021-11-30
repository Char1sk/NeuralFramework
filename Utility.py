import numpy as np
from numba import jit


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
    if funcname == 'hardlim':
        return (hardlim, None)
    if funcname == 'hardlims':
        return (hardlims, None)


# 【激活函数】硬极限函数
def hardlim(x):
    x = np.where(x > 0, 1, 0)
    return x


# 【激活函数】对称硬极限函数
def hardlims(x):
    x = np.where(x > 0, 1, -1)
    return x


# 【激活函数】线性函数
@jit(nopython=True)
def linear(x):
    return x


# 【激活函数导数】线性函数
def dLinear(x):
    return np.ones_like(x)


# 【激活函数】Sigmoid函数
# @jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 【激活函数导数】Sigmoid函数
def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 【损失函数】均方误差函数
def meanSquareError(a, y):
    return 1 / 2 * np.sum((a - y) ** 2)


# 【损失函数导数】均方误差函数
@jit(nopython=True)
def dMeanSquareError(a, y):
    return a - y


# 【损失函数】交叉熵函数(最后一层Softmax激活)
def softmaxCrossEntropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


# 【损失函数导数】交叉熵函数(最后一层Softmax激活)
def dSoftmaxCrossEntropy(a, y):
    return a - y


# 矩阵对应相乘
def elementMultiply(a, b):
    return a*b


# 矩阵叉乘
def matrixMultiply(a, b):
    return np.dot(a, b)


# 更新权重
def updateWeight(w, lr, grad_w):
    w = w-lr*grad_w
    return w


# 【Momentum】更新权重
def updateWeightMomentum(w, lr, vw):
    w = w-lr*vw
    return w


# 【Momentum】更新动量
def updateVW(vw, dr, grad_w):
    vw = dr * vw + (1-dr) * grad_w
    return vw


# 【Adam】更新权重
def updateWeightAdam(w, lr, mtt, vtt):
    w -= lr * mtt / (np.sqrt(vtt) + 1e-8)
    return w


# 【Adam】更新mt
# @jit(nopython=True)
def updateMT(mt, b1, grad_w):
    mt = b1*mt+(1-b1)*grad_w
    return mt


# 【Adam】更新vt
def updateVT(vt, b2, grad_w):
    vt = b2 * vt + (1-b2)*(grad_w**2)
    return vt


# 【Adam】更新mtt
def updateMTT(mt, b1, k):
    mtt = mt/(1-(b1**(k+1)))    # mt的偏置矫正
    return mtt


# 【Adam】更新vtt
def updateVTT(vt, b2, k):
    vtt = vt/(1-(b2**(k+1)))    # vt的偏置矫正
    return vtt
