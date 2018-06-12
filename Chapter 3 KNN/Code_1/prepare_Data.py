import numpy as np

import matplotlib as mpt
import matplotlib.pyplot as plt


def GetData():
    Data = []
    Label = []
    with open(r'C:\Users\source\repos\3_ KNN\datingTestSet2.txt') as fr:
        index = 0
        for line in fr.readlines():
            line = line.strip().split()
            Data.append([float(line[0]), float(line[1]), float(line[2])]) 
            Label.append(int(line[-1]))
            index += 1
        return Data, Label


def Norm(Data):
    minVals = Data.min(0)   #   axis = 0
    maxVals = Data.max(0)
    ranges = maxVals - minVals  #   1X3
    NormData = np.zeros(Data.shape)    #   较大的数组应当提前用占位符提供空间
    m = Data.shape[0]
    NormData = (Data - np.tile(minVals, (m,1))) / (np.tile(ranges, (m, 1)))
    return NormData


