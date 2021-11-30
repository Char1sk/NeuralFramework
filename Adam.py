import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer
from Dataset import Dataset
from Setting import Setting
from Model import Model
from scipy.io import loadmat
import time
# import Utility as ut
import UtilityJit as ut


class Adam(Model):
    # 一般没有什么额外设置，如有则在Setting里添加
    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    def train(self):
        start = time.time()
        train_size = self.trainData.shape[1]
        L = self.depth
        mt, vt = {}, {}

        b1 = 0.9      # 第一次估计的指数衰减率
        b2 = 0.999    # 第二次估计的指数衰次减率
        for l in range(1, L):
            mt[l] = np.zeros((self.layers[l].count, self.layers[l-1].count))   # 梯度一阶矩阵
            vt[l] = np.zeros((self.layers[l].count, self.layers[l-1].count))   # 梯度二阶矩阵
        for epoch_num in range(self.epoch):
            startTime = time.time()
            order = np.random.permutation(train_size)
            for k in range(int(train_size/self.batch)):
                start_idx = k * self.batch
                end_idx = min((k+1)*self.batch, train_size)
                idx = order[start_idx:end_idx]
                self.layers[0].a = self.trainData[:, idx]
                m = len(idx)
                y = self.trainLabel[:, idx]
                # forward computation
                for l in range(0, L-1):
                    self.layers[l+1].z = np.dot(self.weight[l+1], self.layers[l].a)
                    self.layers[l+1].a = self.layers[l].activation(self.layers[l+1].z)
                # backward computation
                self.layers[L-1].delta = self.dCostFunction(self.layers[L-1].a, y) * \
                    self.layers[L-1].dactivation(self.layers[L-1].z)
                for l in range(L - 2, 0, -1):
                    self.layers[l].delta = np.dot(self.weight[l+1].T, self.layers[l+1].delta) \
                                           * self.layers[l].dactivation(self.layers[l].z)
                # weights update
                for l in range(0, L-1):
                    # print(self.layers[l + 1].delta.shape)
                    grad_w = np.dot(self.layers[l + 1].delta, self.layers[l].a.T)/m

                    mt[l+1] = ut.updateMT(mt[l+1], b1, grad_w)
                    vt[l+1] = ut.updateVT(vt[l+1], b2, grad_w)

                    mtt = ut.updateMTT(mt[l+1], b1, k)  # #mt的偏置矫正
                    vtt = ut.updateVTT(vt[l+1], b2, k)  # #vt的偏置矫正

                    # self.weight[l+1] -=    self.alpha * mtt / (np.sqrt(vtt) + 1e-8)
                    self.weight[l+1] = ut.updateWeightAdam(self.weight[l+1], self.alpha, mtt, vtt)

            # train process
            self.trainOutputs.append(self.getOutput(self.trainData))
            # validate process
            self.validateOutputs.append(self.getOutput(self.validateData))
            # test process
            self.testOutputs.append(self.getOutput(self.testData))

            endTime = time.time()

            print("{}/{}: train acc = {:.4f} || validate acc = {:.4f}   time={:.4f}s"
                  .format(epoch_num + 1, self.epoch,
                          self.calculateAccuracy(self.trainOutputs[-1], self.trainLabel),
                          self.calculateAccuracy(self.validateOutputs[-1], self.validateLabel),
                          endTime - startTime))
        # test result
        self.testResult = self.testOutputs[-1]
        # train result
        self.trainResult = self.trainOutputs[-1]
        # validate result
        self.validateResult = self.validateOutputs[-1]

        end = time.time()
        print("globaltime={}s".format(end-start))


if __name__ == '__main__':
    # Data Preparation
    m = loadmat("./mnist_small_matlab.mat")
    trainData, trainLabels = m['trainData'], m['trainLabels']
    testData, testLabels = m['testData'], m['testLabels']
    train_size = 10000
    X_all = trainData.reshape(-1, train_size)
    X_train = X_all[:, :8000]
    Y_train = trainLabels[:, :8000]
    val_size = 2000
    X_validate = X_all[:, 8000:]
    Y_validate = trainLabels[:, 8000:]
    test_size = 2000
    X_test = testData.reshape(-1, test_size)
    Y_test = testLabels

    data = Dataset(trainSet=[X_train, Y_train], validateSet=[X_validate, Y_validate], testSet=[X_test, Y_test])
    l1 = Layer(784, 'sigmoid')
    l2 = Layer(512, 'sigmoid')
    l3 = Layer(256, 'sigmoid')
    l4 = Layer(64, 'sigmoid')
    l5 = Layer(10, 'sigmoid')
    layers = [l1, l2, l3, l4, l5]
    para = Setting(layers=layers, batch=100, epoch=100, alpha=0.005)    # 学习率要偏小些效果才好
    model = Adam(data, para)
    model.train()
    print("Accuracy  = {:<.4f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
    print("Recall    = {}".format(model.calculateRecall(model.trainResult, model.trainLabel)))
    print("Precision = {}".format(model.calculatePrecision(model.trainResult, model.trainLabel)))
    print("F1Score   = {}".format(model.calculateF1Score(model.trainResult, model.trainLabel)))

    plt.grid(axis='y', linestyle='-.')
    plt.plot(np.arange(model.epoch), model.calculateAccuracy(model.trainOutputs, model.trainLabel), label="train", c="b")
    plt.plot(np.arange(model.epoch), model.calculateAccuracy(model.testOutputs, model.testLabel), label="test", c="r")
    plt.plot(np.arange(model.epoch), model.calculateAccuracy(model.validateOutputs, model.validateLabel), label="valid", c="y")
    plt.title("Accuracy")
    plt.legend()
    plt.show()
