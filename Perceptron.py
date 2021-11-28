import numpy as np
from Layer import Layer
import matplotlib.pyplot as plt
from Dataset import Dataset
from Setting import Setting
from Model import Model
#import Utility as ut
import UtilityJit as ut 

class Perceptron(Model):
    # 一般没有什么额外设置，如有则在Setting里添加
    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    # 此处直接从PerceptronTrain复制的
    def train(self):
        print("#### PerceptronTrain Begin ####\n")
        for epoch in range(self.epoch):  #
            print("epoch{:<3d}: w = {}, b = {}".format(epoch, self.weight[1], self.bias))
            error = np.zeros((1, self.trainData.shape[1]))
            for i in range(self.trainData.shape[1]):
                pred = ut.hardlim(np.dot(self.weight[1], self.trainData[:, i:i + 1]) + self.bias)
                error[:, i] = self.trainLabel[:, i:i + 1] - pred

                self.weight[1] += np.dot(error[:, i:i + 1], self.trainData[:, i:i + 1].T)
                self.bias += error[:, i:i + 1]
                self.trainOutputs.append(pred)
                print(pred, "<->", self.trainLabel[:, i:i + 1])

            if error.max() == 0 and error.min() == 0:  # 分类完全正确
                break
        print("\n#### PerceptronTrain End ####")
        pred = np.zeros((self.trainLabel.shape))
        for i in range(self.trainData.shape[1]):
            pred[:, i] = ut.hardlim(np.dot(self.weight[1], self.trainData[:, i:i + 1]) + self.bias)
        self.trainResult = pred


## 这部分属于用户自己操作，我们不用实现
#def draw(data, label):
#    plt.xlim(-2, 2)
#    plt.ylim(-2, 2)
#    plt.scatter(data[0, :], data[1, :], c=label[0, :])
#    x = np.array([-2, 2])
#    y = -(self.weight[1][:, 0] * x + self.bias) / self.weight[1][:, 1]
#    plt.plot(x, y.reshape(2, ), c='r')
#    plt.show()


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
    l1 = Layer(2, 'hardlim')
    l2 = Layer(1, 'hardlim')
    layers = [l1, l2]
    s = Setting(layers=layers)
    model = Perceptron(data, s)
    # 把全部数据都用作模型训练集
    model.trainData = data.allData
    model.trainLabel = data.allLabel
    # 感知机训练
    model.train()
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
    print("Recall    = {}".format(model.calculateRecall(model.trainResult, model.trainLabel)))
    print("Precision = {}".format(model.calculatePrecision(model.trainResult, model.trainLabel)))
    print("F1Score   = {}".format(model.calculateF1Score(model.trainResult, model.trainLabel)))
    #draw(model.trainData, model.trainLabel)    # 画出分类结果
