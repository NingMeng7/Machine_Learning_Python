from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from prepare_Data import img2vector


def KNN(x, Data, Label, k):
    m = Data.shape[0]

    distance = ((tile(x, (m, 1)) - Data)**2).sum(axis = 1)
    distance = distance ** 0.5  #   m x 1
 
    
    sortdistance = distance.argsort()

    classcount = {}
    
    for i in range(k):
        voteLabel = Label[sortdistance[i]]  #   距离第i小的在原数组中的索引值: sortdistance[i]
        classcount[voteLabel] = classcount.get(voteLabel, 0) + 1
    Sortedclasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)
    return Sortedclasscount[0][0]


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(r'C:\Users\xw201\source\repos\3_Knn_\trainingDigits')
    m = len (trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)    # 记录样本真值
        trainingMat[i,:] = img2vector(r'C:\Users\xw201\source\repos\3_Knn_\trainingDigits/%s' % fileNameStr)
    testFileList = listdir(r'C:\Users\xw201\source\repos\3_Knn_\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'C:\Users\xw201\source\repos\3_Knn_\testDigits/%s' % fileNameStr)
        classifierResult = KNN(vectorUnderTest, trainingMat, hwLabels, 3)
        print('The classifier came back with: %d, the real answer is: %d' 
            % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print('The total number of errors is: %d' % errorCount)
    print('The total error rate is: %f' % (errorCount/float(mTest)))

handwritingClassTest()
