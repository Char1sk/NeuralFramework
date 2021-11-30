import numpy as np
from Layer import Layer
from Dataset import Dataset
from Setting import Setting
from Perceptron import Perceptron

if __name__ == '__main__':
    # 二分类感知机测试
    # data为二维坐标，label为0/1
    # 对应模型层数为2，2输入1输出
    data = np.array([[0, 0],
                     [1, 0],
                     [0, 1],
                     [1, 1]])
    label = np.array([[1, 1, 1, 0]])
    # 数据过少不进行分划
    data = Dataset(allSet=[data.T, label])
    # 只需额外定义layers
    s = Setting([Layer(2, 'hardlim'), Layer(1, 'hardlim')], epoch=5)
    model = Perceptron(data, s)
    # 把全部数据都用作模型训练集
    model.trainData = data.allData
    model.trainLabel = data.allLabel
    model.validateData = data.allData
    model.validateLabel = data.allLabel
    model.testData = data.allData
    model.testLabel = data.allLabel
    # 感知机训练
    model.train()
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
    print("Recall    = {}".format(model.calculateRecall(model.trainResult, model.trainLabel)))
    print("Precision = {}".format(model.calculatePrecision(model.trainResult, model.trainLabel)))
    print("F1Score   = {}".format(model.calculateF1Score(model.trainResult, model.trainLabel)))   # 画出分类结果
