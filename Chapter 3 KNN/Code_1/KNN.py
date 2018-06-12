import numpy as np
from prepare_Data import GetData
from prepare_Data import Norm
import operator

Dict = {'Love':3, 'Like':2, 'Hate':1}


def KNN(x, Data, Label, k):
    m = Data.shape[0]

    distance = ((np.tile(x, (m, 1)) - Data)**2).sum(axis = 1)
    distance = distance ** 0.5  #   m x 1
 
    
    sortdistance = distance.argsort()

    classcount = {}
    
    for i in range(k):
        voteLabel = Label[sortdistance[i]]  #   距离第i小的在原数组中的索引值: sortdistance[i]
        classcount[voteLabel] = classcount.get(voteLabel, 0) + 1
    Sortedclasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)
    return Sortedclasscount[0][0]


def datingClassTest():
    TestRate = 0.10  # 测试比例
    Data, Label = GetData()
    Data = np.array(Data)
    #Label = np.array(Label)
    normData = Norm(Data)
    
    m = normData.shape[0]    # 总样本数
    numTestvecs = int(m*TestRate)
    errorCount = 0.0
    for i in range(numTestvecs):
        classifierResult = KNN(normData[i, :], normData[numTestvecs:m,:],
            Label[numTestvecs:m], 7)  # 余下所有样本作为训练集
        if (classifierResult != Label[i]):
            print('The classifier came back with: %d, the real answer is: %d'
                % (classifierResult, Label[i]))
        if (classifierResult != Label[i]):
            errorCount += 1.0
    print('The total error rate is: %f' % (errorCount/float(numTestvecs)))




datingClassTest()
