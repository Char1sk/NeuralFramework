from Dataset import Dataset
from Setting import Setting
# from ForwardNetwork import ForwardNetwork
# from SGD import SGD
# from Momentum import Momentum
from Adam import Adam
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    Labels = np.zeros((10, labels.shape[0]))
    for i in range(0, labels.shape[0]):
        idx = labels[i]
        Labels[idx][i] = 1

    Images = images.T
    return Images, Labels


trainData, trainLabels = load_mnist('mnist/')
testData, testLabels = load_mnist('mnist/', kind='t10k')

train_size = 20000
X_all = trainData
X_train = X_all[:, :train_size]
Y_train = trainLabels[:, :train_size]
val_size = 5000
X_validate = X_all[:, val_size:]
Y_validate = trainLabels[:, val_size:]
test_size = 10000
X_test = testData
Y_test = testLabels
print(trainLabels.shape)
data = Dataset(trainSet=[X_train, Y_train], validateSet=[X_validate, Y_validate], testSet=[X_test, Y_test])
para = Setting()
# para.loadSetting('./testSettingForward.json')
# para.loadSetting('./testSettingSGD.json')
# para.loadSetting('./testSettingMomentum.json')
para.loadSetting('./testSettingAdam.json')
# model = ForwardNetwork(data, para)
# model = SGD(data, para)    # 1    0.01
# model = Momentum(data, para)
model = Adam(data, para)
model.train()


print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
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
