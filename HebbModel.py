import numpy as np
from Dataset import Dataset
from Setting import Setting
from Layer import Layer
from Model import Model
#import Utility as ut
import UtilityJit as ut 

class HebbModel(Model):
    
    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    
    def train(self):
        print("#### HebbTrain Begin ####\n")
        for i in range(self.trainData.shape[1]):
            self.weight[1] += np.dot(self.trainLabel[:, i:i + 1],self.trainData[:, i:i + 1].T)
        print("\n#### HebbTrain End ####")
        pred = np.zeros((self.trainLabel.shape))
        for i in range(self.trainData.shape[1]):
            pred[:, i:i+1] = ut.hardlims(np.dot(self.weight[1], self.trainData[:, i:i + 1]))
        self.trainResult = pred



if __name__ == '__main__':
    data = np.array([[1, -1, 1, 1],
                     [1, 1, -1, 1],
                     [-1, -1, -1, 1]],)
    label = data

    data = Dataset(allSet=[data.T, label.T])
    # s = Setting([4,4],initialize='zero')
    s = Setting([Layer(4,'linear'), Layer(4,'linear')], initialize='zero')
    model = HebbModel(data, s)
    model.trainData = data.allData
    model.trainLabel = data.allLabel

    model.train()
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult, model.trainLabel)))
    print("Recall    = {}".format(model.calculateRecall(model.trainResult, model.trainLabel)))
    print("Precision = {}".format(model.calculatePrecision(model.trainResult, model.trainLabel)))
    print("F1Score   = {}".format(model.calculateF1Score(model.trainResult, model.trainLabel)))
