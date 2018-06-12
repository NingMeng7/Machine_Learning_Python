import numpy as np
from numpy import *
eps = 0.00001
def loadDataSet(fileName):
	dataMat = []
	labelMat = []
	with open(fileName) as fr:
		for line in fr.readlines():
			lineArr = line.strip().split('\t')
			dataMat.append([float(lineArr[0]), float(lineArr[1])])###
			labelMat.append([float(lineArr[2])])
	return dataMat, labelMat


def sel_randj(i, m):	# 生成α对
	j = i
	while (j == i):
		j = int(random.uniform(0, m))
	return j 


def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj


def SMO_simple_version(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = mat(dataMatIn)	# mxn
	labelMat = mat(classLabels).transpose()	# 1xm
	# print(shape(labelMat))
	b = 0
	m, n = shape(dataMatrix)
	alphas = mat(zeros((m, 1)))	#mx1
	iter = 0
	while iter < maxIter:
		alphaParisChanged = 0
		for i in range(m):
			#print((multiply(alphas, labelMat).T)*(dataMatrix*dataMatrix[i,:].T))
			fXi = float(((multiply(alphas.T, labelMat))*(dataMatrix*dataMatrix[i,:].T))[0]) + b 		
			Ei = fXi - labelMat[0,i]
			if((labelMat[0, i]*Ei < -toler) and (alphas[i,0] < C)) or ((labelMat[0, i]*Ei > toler) and (alphas[i,0] > 0)):
				j = sel_randj(i, m)
			fXj = float((multiply(alphas.T, labelMat)*(dataMatrix*dataMatrix[j, :].T))[0]) + b
			Ej = fXj - labelMat[0,j]
			alphaIold = alphas[i, 0].copy()		# 记录
			alphaJold = alphas[j, 0].copy()		# 记录
			if (labelMat[0, i] != labelMat[0, j]):	# 在[0,C]X[0,C] 中 y(1)α1 + y(2)α2 = ξ 得到的H L
				L = max(0, alphas[j, 0] - alphas[i, 0])
				H = min(C, C + alphas[j, 0] - alphas[i, 0])
			else:
				L = max(0, alphas[j, 0] + alphas[i, 0] - C)
				H = min(C, alphas[j, 0] + alphas[i, 0])
			if L == H:
				print('L==H!')
				continue
			eta = 2.0 * dataMatrix[i, :] * dataMatrix[i, :].T - \
				dataMatrix[i, :] * dataMatrix[i, :].T - \
					dataMatrix[j, :] * dataMatrix[j, :].T
			if eta >=0:
				print('eta>=0!')
				continue
			alphas[j, 0] = alphas[j, 0] - labelMat[0, j] * (Ei - Ej) / eta
			alphas[j, 0] = clipAlpha(alphas[j, 0], H, L)
			if (abs(alphas[j, 0] - alphaJold) < eps):
				print('j not moving enough!')
				continue
			alphas[i, 0] += labelMat[0, j] * labelMat[0, i] * (alphaJold - alphas[j, 0])
			b1 = b - Ei - labelMat[0, i] * (alphas[i, 0] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
				labelMat[0, j] * (alphas[j, 0] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
			b2 = b - Ej - labelMat[0, i] * (alphas[i, 0] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
				labelMat[0, j] * (alphas[j, 0] - alphaJold) * dataMatrix[j, :]*dataMatrix[j, :].T
			if (0 < alphas[i, 0]) and (C > alphas[j, 0]):
				b = b1
			elif (0 < alphas[j, 0]) and (C > alphas[j, 0]):
				b = b2
			else:
				b = (b1 + b2) / 2.0
			alphaParisChanged += 1
			print('Iter: %d i: %d, pairschanged %d' % (iter, i, alphaParisChanged))
		if (alphaParisChanged == 0):
			iter += 1
		else:
			iter = 0
		print('Iteration number: %d' % iter)
	return b, alphas

def testRbf():
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = SMO_simple_version(dataArr, labelArr, 0.6, 0.0001, 40) #C=200 important
    dataMatrix=mat(dataArr); 
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMatrix)
    errorCount = 0
    for i in range(m):
        predict = float(((multiply(alphas.T, labelMat))*(dataMatrix*dataMatrix[i,:].T))[0]) + b 		
        if sign(predict)!=sign(labelMat[0, i]): 
            errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))   
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMatrix=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(dataMatrix)
    for i in range(m):
        predict = float(((multiply(alphas.T, labelMat))*(dataMatrix*dataMatrix[i,:].T))[0]) + b 		
        if sign(predict)!=sign(labelMat[0, i]): 
            errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))    
    
#dataArr, labelArr = loadDataSet('testSet.txt')
##print(dataArr)
##print(labelArr)
#b, alphas = SMO_simple_version(dataArr, labelArr, 0.6, 0.001, 40)
#print(b)
#print(alphas)
testRbf()