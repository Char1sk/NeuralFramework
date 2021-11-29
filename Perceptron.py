import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer
from Dataset import Dataset
from Setting import Setting
from Model import Model
from scipy.io import loadmat
import math,time
#import Utility as ut 
import UtilityJit as ut 

class Perceptron(Model):
    # 一般没有什么额外设置，如有则在Setting里添加
    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    # 此处直接从PerceptronTrain复制的
    def train(self):
        start = time.time()
        train_size = self.trainData.shape[1]
        for epoch_num in range(self.epoch):  
            
            idxs = np.random.permutation(train_size)
            right = 0    
            
            startTime = time.time()
            for k in range(math.ceil(train_size/self.batch)):      
                start_idx = k*self.batch 
                end_idx = min((k+1)*self.batch, train_size)           
                batch_indices = idxs[start_idx:end_idx]

                self.layers[0].a = self.trainData[:, batch_indices]
                y = self.trainLabel[:, batch_indices]
                
                self.layers[1].a=ut.hardlim(np.dot(self.weight[1], self.layers[0].a)+self.bias)
                error=y-self.layers[1].a
                #loss+=np.sum(error**2)            
                
                grad_w = -np.dot(error, self.layers[0].a.T)
                self.weight[1] = ut.updateWeight(self.weight[1], self.alpha, grad_w) 
                self.bias = self.bias + self.alpha*np.sum(error,axis=1).reshape(10,1)

            # train process
            self.trainOutputs.append(self.getOutput(self.trainData))
            # validate process
            self.validateOutputs.append(self.getOutput(self.validateData))
            # test process
            self.testOutputs.append(self.getOutput(self.testData))

            endTime = time.time()

            print("{}/{}: train acc = {:.4f} || validate acc = {:.4f}   time={:.4f}s"\
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
    l1 = Layer(784, 'hardlim')
    l2 = Layer(10, 'hardlim')
    layers = [l1, l2]
    para = Setting(layers=layers, batch=100, epoch=100, alpha=0.01)
    model = Perceptron(data, para)
    model.train()
    
    #print(model.calculateAccuracy(model.trainOutputs, model.trainLabel))
    print("Accuracy  = {:<.4f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
    print("Recall    = {}".format(model.calculateRecall(model.trainResult, model.trainLabel)))
    print("Precision = {}".format(model.calculatePrecision(model.trainResult, model.trainLabel)))
    print("F1Score   = {}".format(model.calculateF1Score(model.trainResult, model.trainLabel)))

    plt.grid(axis='y',linestyle='-.')
    plt.plot(np.arange(model.epoch),model.calculateAccuracy(model.trainOutputs, model.trainLabel),label="train",c="b")
    plt.plot(np.arange(model.epoch),model.calculateAccuracy(model.testOutputs, model.testLabel),label="test",c="r")
    plt.plot(np.arange(model.epoch),model.calculateAccuracy(model.validateOutputs, model.validateLabel),label="valid",c="y")
    plt.title("Accuracy")
    plt.legend()
    plt.show()