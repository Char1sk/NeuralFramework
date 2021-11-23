# from Model import Model
import Dataset
import numpy as np
import random
import Utility as ut

# alldata = np.zeros((10,7))
# for i in range(7):
#     alldata[i, i] = 1
alllabel = np.array([[1,2,3,4,5,6,7,8,9,10]])
# print(alldata)
# p =random.shuffle(alldata)
# print()
# print(p)
# print(np.random.permutation(10))

# def hardlim(x):
#     x = np.where(x > 0, 1, 0)
#     return x
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

func_list = ['ut.hardlim', 'ut.sigmoid']
print(eval(func_list[1])(alllabel))
