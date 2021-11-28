import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
from HebbModel import HebbModel
from Dataset import Dataset
from Setting import Setting
from Layer import Layer


def testZero(WeightMatrix):  # 用0图像上半部分进行测试
    test0 = [-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1]
    for i in range(0, 15):
        test0.append(-1)
    arrayTest0 = np.array(test0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(arrayTest0.size):
        j = i % 5
        if (arrayTest0[i] == -1):
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='white')
        else:
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='black')
        ax.add_patch(rect)
    plt.xlim([0, 50])
    plt.ylim([0, 60])
    plt.show()
    MartixTest0 = arrayTest0.reshape(arrayTest0.size, 1)
    output = np.matmul(WeightMatrix, MartixTest0)
    for i in range(output.size):
        if (output[i] >= 0):
            output[i] = 1
        else:
            output[i] = -1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(output.size):
        j = i % 5
        if (output[i] == -1):
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='white')
        else:
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='black')
        ax.add_patch(rect)
    plt.xlim([0, 50])
    plt.ylim([0, 60])
    plt.show()

def testOne(WeightMatrix):  # 用1图像下半部分进行测试
    test1 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1]
    arrayTest1 = np.array(test1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(arrayTest1.size):
        j = i % 5
        if (arrayTest1[i] == -1):
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='white')
        else:
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='black')
        ax.add_patch(rect)
    plt.xlim([0, 50])
    plt.ylim([0, 60])
    plt.show()
    MartixTest1 = arrayTest1.reshape(arrayTest1.size, 1)
    output = np.matmul(WeightMatrix, MartixTest1)
    for i in range(output.size):
        if (output[i] >= 0):
            output[i] = 1
        else:
            output[i] = -1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(output.size):
        j = i % 5
        if (output[i] == -1):
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='white')
        else:
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='black')
        ax.add_patch(rect)
    plt.xlim([0, 50])
    plt.ylim([0, 60])
    plt.show()

def testTwo(WeightMatrix):  # 用2图像一部分进行测试
    test2 = [1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1]
    arrayTest2 = np.array(test2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(arrayTest2.size):
        j = i % 5
        if (arrayTest2[i] == -1):
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='white')
        else:
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='black')
        ax.add_patch(rect)
    plt.xlim([0, 50])
    plt.ylim([0, 60])
    plt.show()
    MartixTest2 = arrayTest2.reshape(arrayTest2.size, 1)
    output = np.matmul(WeightMatrix, MartixTest2)
    for i in range(output.size):
        if (output[i] >= 0):
            output[i] = 1
        else:
            output[i] = -1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(output.size):
        j = i % 5
        if (output[i] == -1):
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='white')
        else:
            rect = matplotlib.patches.Rectangle((0 + j * 10, 50 - math.floor(i / 5) * 10), 10, 10, color='black')
        ax.add_patch(rect)
    plt.xlim([0, 50])
    plt.ylim([0, 60])
    plt.show()




data = np.array([[-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
                [-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
                [1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1]],)
label = data

data = Dataset(allSet=[data.T, label.T])
s = Setting([Layer(30, 'linear'), Layer(30, 'linear')], initialize='zero')
model = HebbModel(data, s)
model.trainData = data.allData
model.trainLabel = data.allLabel
model.train()
WeightMatrix = model.weight[1]

testZero(WeightMatrix)
testOne(WeightMatrix)
testTwo(WeightMatrix)