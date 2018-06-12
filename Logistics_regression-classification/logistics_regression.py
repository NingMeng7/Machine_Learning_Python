# matrix.getA() -> array 似乎python3不能用了
# matrix: a*b:矩阵乘法 a.T:a' a.H:共轭a a.I:逆矩阵
# matrix: a**2:矩阵乘幂
# array: 运算是对每个元素进行运算  a**2 || a*a: 每个元素进行平方
# array: 矩阵乘法要调用np.dot来实现，只有a.T
# plt.figure().add_subplot(m,n,p) 把画布分割成mxn，并在按照左到右，上到下的次序的第p块上绘图

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


def gradAscent(dataMatIn, classLabels):	# 梯度下降
	dataMatrix = np.mat(dataMatIn)	# 100X3 n=2 n+1=3
	labelMat = np.mat(classLabels).transpose()	# 变成列向量
	m, n = np.shape(dataMatrix)
	
	alpha = 0.001
	maxCycles = 500

	weights = np.ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)	# mX1 a(i,1) = θ'X^(i)  
		differ = labelMat - h # differ m*1: y^(i) - h(x^(i))
		differ *= -1
		weights = weights - alpha * dataMatrix.transpose() * differ # 求梯度并更新
	return weights



def stocGradDescent(dataMatrix, classLabels, numIter=150):
	m, n = np.shape(dataMatrix)
	weights = np.ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i) + 0.01 #
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			error *= -1
			weights = weights - alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights


dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
weights_ = stocGradDescent(array(dataArr), labelMat)
print(weights)
plotBoundary(array(weights))
input('wating...')
plotBoundary(array(weights_))
print(weights_)






















