import numpy as np
from Dataset import Dataset
from Setting import Setting
from Layer import Layer
from Model import Model
# import Utility as ut
import UtilityJit as ut
import matplotlib
import matplotlib.pyplot as plt
import math


class HebbModel(Model):

    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    def train(self):
        print("#### HebbTrain Begin ####\n")
        for i in range(self.trainData.shape[1]):
            self.weight[1] += np.dot(self.trainLabel[:, i:i + 1], self.trainData[:, i:i + 1].T)
        print("\n#### HebbTrain End ####")
        pred = np.zeros((self.trainLabel.shape))
        for i in range(self.trainData.shape[1]):
            pred[:, i:i+1] = ut.hardlims(np.dot(self.weight[1], self.trainData[:, i:i + 1]))
        self.trainResult = pred


def draw(trainResult):
    for k in range(trainResult.shape[1]):
        output = trainResult[:, k]
        for i in range(output.size):
            if(output[i] >= 0):
                output[i] = 1
            else:
                output[i] = -1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(output.size):
            j = i % 2
            if(output[i] == -1):
                rect = matplotlib.patches.Rectangle((0+j*10, 10-math.floor(i/2)*10), 10, 10, color='yellow')
            else:
                rect = matplotlib.patches.Rectangle((0+j*10, 10-math.floor(i/2)*10), 10, 10, color='red')
            ax.add_patch(rect)
        plt.xlim([0, 20])
        plt.ylim([0, 20])
        plt.show()


if __name__ == '__main__':
    data = np.array([[1., -1., 1., 1.],
                     [1., 1., -1., 1.],
                     [-1., -1., -1., 1.]],)
    label = data

    data = Dataset(allSet=[data.T, label.T])
    # s = Setting([4,4],initialize='zero')
    s = Setting([Layer(4, 'linear'), Layer(4, 'linear')], initialize='zero')
    model = HebbModel(data, s)
    model.trainData = data.allData
    model.trainLabel = data.allLabel

    model.train()
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
    draw(model.trainResult)
