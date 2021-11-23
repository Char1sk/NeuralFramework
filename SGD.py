import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset
from Setting import Setting
from Model import Model
from Utility import sigmoid, sigmoidprop, hardlim


class SGD(Model):
    # 一般没有什么额外设置，如有则在Setting里添加
    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    def train(self):
        train_size = self.trainData.shape[1]
        L = self.depth
        for epoch_num in range(self.epoch):
            idxs = np.random.permutation(train_size)
            for k in range(idxs):       # SGD形式，每次只选择一个样本求更新结果
                self.layers[0].a = self.trainData[:, k]
                y = self.trainLabel[:, k]
                # forward computation
                for l in range(0, L-1):
                    self.layers[l+1].z = np.dot(self.weight[l], self.layers[l].a)
                    self.layers[l+1].a = eval(self.layers[l].activation)(self.layers[l+1].z)
                # backward computation
                self.layers[L-1].delta = (self.layers[L-1].a - y) * \
                                         eval(self.layers[L-1].activation_prop)(self.layers[L-1].z)
                for l in range(L - 2, 0, -1):
                    self.layers[l].delta = np.dot(self.weight[l].T, self.layers[l+1].delta) \
                                           * eval(self.layers[l].activation_prop)(self.layers[l].z)
                # weights update
                for l in range(0, L-1):
                    grad_w = np.dot(self.layers[l + 1].delta, self.layers[l].a.T)
                    self.weight[l] = self.weight[l] - self.alpha * grad_w
            # train process
            self.layers[0].a = self.trainData
            for l in range(0, L - 1):
                self.layers[l + 1].z = np.dot(self.weight[l], self.layers[l].a)
                self.layers[l + 1].a = eval(self.layers[l].activation)(self.layers[l + 1].z)
            self.trainOutputs.append(self.layers[L-1].a)
            # validate process
            self.layers[0].a = self.validateData
            for l in range(0, L - 1):
                self.layers[l + 1].z = np.dot(self.weight[l], self.layers[l].a)
                self.layers[l + 1].a = eval(self.layers[l].activation)(self.layers[l + 1].z)
            self.validateOutputs.append(self.layers[L - 1].a)
        # test result
        self.layers[0].a = self.testData
        for l in range(0, L - 1):
            self.layers[l + 1].z = np.dot(self.weight[l], self.layers[l].a)
            self.layers[l + 1].a = eval(self.layers[l].activation)(self.layers[l + 1].z)
        self.testResult = self.layers[L-1].a
        # train result
        self.trainResult = self.trainOutputs[-1]
        # validate result
        self.validateResult = self.validateOutputs[-1]
