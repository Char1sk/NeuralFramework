# from Model import Model
import numpy as np
import pickle
from Dataset import Dataset
from Setting import Setting
from Model import Model

if __name__ == '__main__':
    # 二分类感知机测试
    # data为二维坐标，label为0/1
    # 对应模型层数为2，2输入1输出
    
    data = np.array([[0,0],
                        [1,0],
                        [0,1],
                        [1,1]])
    label = np.array([[1, 1, 1, 0]])
    
    # 数据过少不进行分划
    data = Dataset(allSet=[data.T, label])
    #data.divideData(1,0)
    #data.showall()

    # 只需额外定义layers
    s = Setting([2,1])
    #s.ParamShow()

    model = Model(data, s)

    # 把全部数据都用作模型训练集
    model.trainData=data.allData
    model.trainLabel=data.allLabel

    #print(model.trainData,model.trainLabel)
    
    model.PerceptronTrain() # 感知机训练
    print("Accuracy  = {:<4.2f}".format(model.calculateAccuracy(model.trainResult,model.trainLabel)),\
          #"Recall    = {:<4.2f}".format(model.calculateRecall(model.trainResult,model.trainLabel)),\       
          #"Precision = {:<4.2f}".format(model.calculatePrecision(model.trainResult,model.trainLabel)),\  
          #"F1Score   = {:<4.2f}".format(model.calculateF1Score(model.trainResult,model.trainLabel))    
          )
    model.draw(model.trainData,model.trainLabel)    # 画出分类结果
    