import numpy as np
from numpy import *
import scipy.optimize as opt
import matplotlib.pyplot as plt


def sigmoid(z):
	g = 1 / (1+np.exp(-z))
	return g

def plotBoundary(wei):
	weights = array(wei)
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)	# 为了进行array运算
	
	n = shape(dataArr)[0]
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
	x = arange(-3.0, 3.0, 0.1)	# arange(start, stop, step)
	y = (-weights[0] - weights[1] * x) / weights[2]	# 蕴含了θ'X=0(边界线g(0)=0.5)
	ax.plot(x, y)
	
	plt.xlabel('X1')
	plt.ylabel('X2')

	plt.show()



def loadDataSet():
	dataMat = []
	labelMat = []
	with open('testSet.txt') as fr:
		for line in fr.readlines():
			lineArr = line.strip().split()
			dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 1.0: x特征值向量是n+1维的
			labelMat.append(int(lineArr[2]))
		return dataMat, labelMat


def stocGradDescent(dataMatrix, classLabels, numIter=150):
	m, n = np.shape(dataMatrix)
	weights = np.ones(n)
	lmd = 0.001
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(10+j+i) + 0.001 #
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))	# array元素做乘法,计算log(hΘ)
			error = h - classLabels[randIndex]
			weights = weights - alpha * error * dataMatrix[randIndex] - alpha * lmd * weights
	return weights



dataMat, labelMat = loadDataSet()
weights = stocGradDescent(array(dataMat), labelMat)
plotBoundary(weights)
print(weights)



