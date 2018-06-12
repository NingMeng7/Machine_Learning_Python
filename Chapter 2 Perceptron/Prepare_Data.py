import numpy as np
import matplotlib.pyplot as plt


def GetData():  # 对数据的输入格式有一定假设
    Data = []
    Label = []
    with open(r'C:\Users\xw201\source\repos\PythonApplication2\testSet.txt') as fr:
        for line in fr.readlines():
            lineArray = line.strip().split()
            Data.append([1.0, float(lineArray[0]), float(lineArray[1])])    #   [1.0, x1, x2]
            Label.append(int(lineArray[2]))    
        return Data, Label


def Plot(Data, Label, weight):
    weights = np.array(weight)
    m = Data.shape[0]
    xrecord1 = []
    yrecord1 = []
    xrecord2 = []
    yrecord2 = []
    for i in range(m):
        if int(Label[i] == 1):
            xrecord1.append(Data[i, 1])
            yrecord1.append(Data[i, 2])
        else:
            xrecord2.append(Data[i, 1])
            yrecord2.append(Data[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xrecord1, yrecord1, s=30, c='red', marker='s')
    ax.scatter(xrecord2, yrecord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)	# arange(start, stop, step)
    y = (-weights[0] - weights[1] * x) / weights[2]	# 蕴含了θ'X=0(边界线g(0)=0.5)
    ax.plot(x, y)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    plt.show()
