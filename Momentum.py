import numpy as np
from Layer import Layer
from Dataset import Dataset
from Setting import Setting
from Model import Model
from scipy.io import loadmat
import math

def accuracy(a, y):
    mini_batch = a.shape[1]
    idx_a = np.argmax(a, axis=0)
    idx_y = np.argmax(y, axis=0)
    acc = sum(idx_a == idx_y) / mini_batch
    return acc


class Momentum(Model):
    # 一般没有什么额外设置，如有则在Setting里添加
    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    def train(self):
        train_size = self.trainData.shape[1]
        L = self.depth
        vw = {}         #Momentum
        dr = 0.9
        for i in range(1, L):
            vw[i] = np.zeros(self.weight[i].shape)

            #print(vw[i].shape)

        for epoch_num in range(self.epoch):
            idxs = np.random.permutation(train_size)

            for k in range(math.ceil(train_size/self.batch)):
                start_idx = k*self.batch 
                end_idx = min((k+1)*self.batch, train_size)
                batch_indices = idxs[start_idx:end_idx]

                self.layers[0].a = self.trainData[:, batch_indices]
                y = self.trainLabel[:, batch_indices]

                #print(self.layers[0].a.shape, y.shape)

                # forward computation
                for l in range(0, L-1):
                    self.layers[l+1].z = np.dot(self.weight[l+1], self.layers[l].a)
                    # self.layers[l+1].a = eval(self.layers[l].activation)(self.layers[l+1].z)
                    self.layers[l+1].a = self.layers[l].activation(self.layers[l+1].z)
                    # print(l, self.layers[l+1].a.shape)

                # backward computation
                self.layers[L-1].delta = self.dCostFunction(self.layers[L-1].a, y) * \
                    self.layers[L-1].dactivation(self.layers[L-1].z)
                # print(self.layers[L-1].delta.shape)

                for l in range(L - 2, 0, -1):
                    self.layers[l].delta = np.dot(self.weight[l+1].T, self.layers[l+1].delta) \
                                           * self.layers[l].dactivation(self.layers[l].z)
                # weights update
                for l in range(0, L-1):

                    grad_w = np.dot(self.layers[l + 1].delta, self.layers[l].a.T)
                    vw[l+1] = dr * vw[l+1] + (1-dr) * grad_w
                    self.weight[l+1] = self.weight[l+1] - self.alpha * vw[l+1]
                    #self.weight[l+1] = self.weight[l+1] - self.alpha * grad_w
                #print("\r{}/{}, acc={}".format(k, math.ceil(train_size/self.batch), accuracy(self.layers[L-1].a, y)),end='')

            # train process
            self.trainOutputs.append(self.getOutput(self.trainData))
            # validate process
            self.validateOutputs.append(self.getOutput(self.validateData))
            print('\n%d/%d train acc: %.4f | validate acc: %.4f' %
                  (epoch_num + 1, self.epoch, self.calculateAccuracy(self.trainOutputs[-1], self.trainLabel),
                   self.calculateAccuracy(self.validateOutputs[-1], self.validateLabel)))
        # test result
        self.testResult = self.getOutput(self.testData)
        # train result
        self.trainResult = self.trainOutputs[-1]
        # validate result
        self.validateResult = self.validateOutputs[-1]


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
    para = Setting(layers=layers, batch=100, epoch=100, alpha=0.01)
    model = Momentum(data, para)
    model.train()
    print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
    print("Recall    = {}".format(model.calculateRecall(model.trainResult, model.trainLabel)))
    print("Precision = {}".format(model.calculatePrecision(model.trainResult, model.trainLabel)))
    print("F1Score   = {}".format(model.calculateF1Score(model.trainResult, model.trainLabel)))

