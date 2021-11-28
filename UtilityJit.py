import numpy as np
from numba import jit,vectorize, float64
import time


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

# 【激活函数】硬极限函数
@jit(nopython=True)
def hardlim(x):
    x = np.where(x > 0, 1, 0)
    return x


# 【激活函数】对称硬极限函数
@jit(nopython=True)
def hardlims(x):
    x = np.where(x > 0, 1, -1)
    return x


# 【激活函数】线性函数
@jit(nopython=True)
def linear(x):
    return x


# 【激活函数导数】线性函数
@jit(nopython=True)
def dLinear(x):
    return np.ones_like(x)


# 【激活函数】Sigmoid函数
@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



# 【激活函数导数】Sigmoid函数
@jit(nopython=True)
def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 【损失函数】均方误差函数
@jit(nopython=True)
def meanSquareError(a, y):
    return 1 / 2 * np.sum((a - y) ** 2)


# 【损失函数导数】均方误差函数
@jit(nopython=True)
def dMeanSquareError(a, y):
    return a - y


# 【损失函数】交叉熵函数(最后一层Softmax激活)
@jit(nopython=True)
def softmaxCrossEntropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


# 【损失函数导数】交叉熵函数(最后一层Softmax激活)
@jit(nopython=True)
def dSoftmaxCrossEntropy(a, y):
    return a - y

# 矩阵对应元素相乘
@jit(nopython=True)
def elementMultiply(a, b):
    return a*b

# 矩阵叉乘
@jit(nopython=True)
def matrixMultiply(a, b):
    return np.dot(a, b)

# 更新权重
@jit(nopython=True)
def updateWeight(w, lr, grad_w):
    w=w-lr*grad_w
    return w 

# 【Momentum】更新权重
@jit(nopython=True)
def updateWeightMomentum(w, lr, vw):
    w=w-lr*vw
    return w

# 【Momentum】更新动量
@jit(nopython=True)
def updateVW(vw, dr, grad_w):
    vw = dr * vw + (1-dr) * grad_w
    return vw

# 【Adam】更新权重
@jit(nopython=True)
def updateWeightAdam(w, lr, mtt, vtt):
    w -= lr * mtt / (np.sqrt(vtt) + 1e-8)  
    return w

# 【Adam】更新mt  
@jit(nopython=True)
def updateMT(mt, b1,grad_w):
    mt  = b1 *mt+(1-b1)*grad_w
    return mt

# 【Adam】更新vt 
@jit(nopython=True) 
def updateVT(vt, b2, grad_w):
    vt = b2 *vt + (1-b2)*(grad_w**2)
    return vt

# 【Adam】更新mtt  
@jit(nopython=True)
def updateMTT(mt, b1, k):
    mtt = mt/(1-(b1**(k+1)))###mt的偏置矫正
                    
    return mtt

# 【Adam】更新vtt 
@jit(nopython=True) 
def updateVTT(vt, b2, k):
    vtt = vt/(1-(b2**(k+1)))##vt的偏置矫正
    return vtt

if __name__ == '__main__':
    arr1=np.random.randn(500,600)
    arr2=np.random.randn(600,700)


    start1=time.time()
    for i in range(5000):
        arr3=np.dot(arr1,arr2)
    end1=time.time()
    print("{:.6f}s".format(end1-start1))


    start2=time.time()
    for i in range(5000):
        arr4=matrixMultiply(arr1,arr2)
    end2=time.time()
    print("{:.6f}s".format(end2-start2))

   
