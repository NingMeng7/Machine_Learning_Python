import numpy as np
import matplotlib.pyplot as plt


def GetData():
    '''
    descripti: 加载数据集
    Args: void
    return: 返回两个array数组
        Data: 数据集的特征向量组成的数组,扩展n+1维为1
        Label: 数据集的标签(类别)
    '''
    
    Data = []
    Label = []
    with open(r'C:\Users\xw201\source\repos\7_logistics_regression_\testSet.txt') as fp:
        for line in fp.readlines():
            line_arr = line.strip().split()
            Data.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
            Label.append(int(line_arr[2]))

    return np.array(Data), np.array(Label)


def sigmoid(x):
    '''
    descript: sigmoid 函数
    Arg: 自变量x
    return: y
    '''
    
    return 1.0 / (1+np.exp(-x))


def grad_ascent(Data, Label):
    '''
    description: 梯度上升法实现极大似然估计 
    Args:  Data and Label
    return: 返回似然估计出的参数 w
    '''
    m, n = Data.shape
    alpha = 0.001
    max_Iterations = 1000
    weights = np.ones((n,1))
    Label = Label.reshape(100,1)

    for k in range(max_Iterations):
        h = sigmoid(np.dot(Data, weights))  # 得到一个m*1的矩阵,每个分量是sigmoid(w·x_i)
        error = Label - h # error (mx1): y^(i) - sigmoid(w·x_i)   #   m * 1
        error *= -1
        weights = weights - alpha * np.dot(Data.T,error) # 求梯度并更新
    return weights


def plotBoundary(wei):
	weights = np.array(wei)
	dataArr, labelMat = GetData()
	
	n = dataArr.shape[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:	# 监督学习有真值哦
			xcord1.append(dataArr[i, 1]) 
			ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1])	# x
			ycord2.append(dataArr[i, 2])	# y
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = np.arange(-5.0, 5.0, 0.1)	# arange(start, stop, step)
	y = (-weights[0] - weights[1] * x) / weights[2]	# 蕴含了θ'X=0(边界线g(0)=0.5)
	ax.plot(x, y)
	
	plt.xlabel('X1')
	plt.ylabel('X2')

	plt.show()
