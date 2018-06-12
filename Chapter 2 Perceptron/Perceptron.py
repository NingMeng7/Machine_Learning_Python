import numpy as np
import matplotlib as mtp
import matplotlib.pyplot as plt
from module1 import Plot
from module1 import GetData


alpha = 0.01   # learning rate



def PLA(Data, Label, MaxIteration):
    m, n = Data.shape
    weights = np.zeros(n)
    flag = 1    #   flag 表示是否找到完全分开数据的超平面
    count = 0
    while flag == 1:
        flag = 0
        count = count + 1
        for i in range(m-1):
            if Label[i] * np.dot(Data[i], weights.T) <= 0: #   如果第i个样本被误分类 这里应该有一个地方要转置
                diff =  np.array(alpha * Label[i] * Data[i])
                weights = weights + diff
                flag = 1
        if count > MaxIteration:
            break;
    return weights




#file = input('Please enter the file name:')
Data, Label = GetData()
Data = np.array(Data)
Label = np.array(Label)
weights = PLA(Data, Label, 1000)
print(weights)
Plot(Data, Label, weights)
