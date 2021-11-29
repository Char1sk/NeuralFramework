import numpy as np
from SGD import SGD
from scipy.io import loadmat
from Layer import Layer
from Dataset import Dataset
from Setting import Setting
import matplotlib.pyplot as plt

def rgbtogray(x):
    r=x[:,:,2,:]
    g=x[:,:,1,:]
    b=x[:,:,0,:]
    y = (0.299*r+0.587*g+0.114*b)/255
    return y

m = loadmat(r"svhn_train_32x32.mat")
tData, tLabels = m['X'], m['y']
trainLabels=np.zeros((10,len(tLabels)))
for i in range(len(tLabels)) :
    t = [0,0,0,0,0,0,0,0,0,0]
    if tLabels[i] == 10:
        t[0]=1
    else:
        t[int(tLabels[i])]=1
    trainLabels[:,i]=t
trainData = rgbtogray(tData)

m = loadmat(r"svhn_test_32x32.mat")
tData, tLabels = m['X'], m['y']
testLabels=np.zeros((10,len(tLabels)))
for i in range(len(tLabels)) :
    t = [0,0,0,0,0,0,0,0,0,0]
    if tLabels[i] == 10:
        t[0]=1
    else:
        t[int(tLabels[i])]=1
    testLabels[:,i]=t
testData = rgbtogray(tData)


train_all_size = 73257
X_all = trainData.reshape(-1, train_all_size)
train_size = train_all_size*0.8
X_train = X_all[:, :int(train_size)]
Y_train = trainLabels[:, :int(train_size)]
X_validate = X_all[:, int(train_size):]
Y_validate = trainLabels[:, int(train_size):]

test_size = 26032
X_test = testData.reshape(-1, test_size)
Y_test = testLabels
print("######################################################")
data = Dataset(trainSet=[X_train, Y_train], validateSet=[X_validate, Y_validate], testSet=[X_test, Y_test])
# 使用代码设置setting
# l1 = Layer(1024, 'sigmoid')
# l2 = Layer(256, 'sigmoid')
# l3 = Layer(10, 'sigmoid')
# layers = [l1, l2, l3]
# para = Setting(layers=layers, batch=1, epoch=10,costFunction='crossEntropy')

para = Setting()
para.loadSetting(r"svhnSetting.json")

model = SGD(data, para)
model.train()
print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
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