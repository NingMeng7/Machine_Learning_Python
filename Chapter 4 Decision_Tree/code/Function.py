import numpy as np
import operator
from math import log

def GetData():
    '''
    description: 从文本文件中读取数据,注意label根据实际问题进行修改
    Args: void
    returns: Data and label
    
    '''
    with open(r'C:\Users\...\...\...\4_Desicion_Tree\lenses.txt') as fr:
        Data = [line.strip().split('\t') for line in fr.readlines()]
        Label = ['age', 'prescript', 'astigmatic', 'tearRate']  #   Label根据实际问题进行修改
        return Data, Label


def cal_entropy(Data):
    '''
    description: 这个函数基于数据集Data计算经验熵熵
    Args: 数据集Data
    returns: empirical entropy
    
    '''
    m = len(Data)
    LabelCounts = {}    # 不同种类的样本数量的计数器
    
    for sample in Data: # 记录类别数量k,以及每一个种类K的样本数目
        currentLabel = sample[-1]
        if currentLabel not in LabelCounts.keys():
            LabelCounts[currentLabel] = 0
        LabelCounts[currentLabel] += 1

    entropy = 0.0
    
    for key in LabelCounts:
        temp = float(LabelCounts[key]/m) 
        entropy -= temp* log(temp, 2)
    
    return entropy


def splitData(Data, axis, value):
    '''
    description: 这个函数将Data中所有:(满足第axis个特征取值为value的样本)删除掉第axis个特征后,划分到D_i中
    Args: Data: 数据集 axis: 特征编号 value: 特征取值
    return: D_i
    
    '''
    subData = [] 
    for sample in Data:
        if sample[axis] == value:
            newsample = sample[:axis]
            newsample.extend(sample[axis+1:])

            subData.append(newsample)
    
    return subData


def choose_best_feature(Data, threshold):  
    '''
    description: 这个函数计算各个特征值的信息增益,来决定出最佳特征
    Args: 数据集Data, 信息增益阈值threshold
    returns: 最佳决策特征的index
    '''
    
    # H(D)
    # H(Y|A) 子集关于子集概率极大似然估计的加权熵
    # g(D,A) = H(D) - H(Y|A)
    oldEntropy = cal_entropy(Data)
    n = len(Data[0]) - 1    # n是特征的数目,最后一列是label所以-1
    
    maxInforGain, BestFeature = 0.0, -1

    # 遍历每一个feature,找到信息增益最大的那个feature
    for feature in range(n):
        featurevalue = [sample[feature] for sample in Data]
        uniqueValues = set(featurevalue)    # 上一句把所有样本的这个特征的取值存了一个list,这里利用集合的互异性去掉了重复的值
        
        newEntropy = 0.0
        for value in uniqueValues:
            subData = splitData(Data, feature, value) # 划分的一个子集D_i,如果读不懂你应该去看看配套笔记里对算法的描述
            probility = len(subData)/float(len(Data))   # 子集概率的极大似然估计
            newEntropy += probility * cal_entropy(subData)

        InfoGain = oldEntropy - newEntropy
        if(InfoGain > maxInforGain):
            maxInforGain = InfoGain
            BestFeature = feature
    
    if maxInforGain < threshold:    # 如果信息增益小于阈值
        return -2
    
    return BestFeature


def majority(classlist):
    '''
    description: 多数表决原则选出出现次数最多的结果
    Args: label 列的集合
    returns: 出现次数最多的结果
    '''
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedclassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) 
        #   以每个标签出现的频率为排序依据降序排序
    return sortedclassCount[0][0]

   
