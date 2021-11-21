import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset
from Setting import Setting
from Model import Model
import Utility as ut


class HebbModel(Model):
    # 一般没有什么额外设置，如有则在Setting里添加
    def __init__(self, dataset, setting):
        super().__init__(dataset, setting)

    # 此处直接从PerceptronTrain复制的
    def train(self):
        # Do training
        pass


# 可以作为用户，进行一些检验训练情况的操作，比如画图
if __name__ == '__main__':
    # Do some test
    pass
