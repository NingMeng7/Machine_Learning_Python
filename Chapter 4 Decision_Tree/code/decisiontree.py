from Functions import GetData
from Functions import choose_best_feature
from Functions import splitData

def CreateTree(Data, Label):
    '''
    description: 创建决策树 by ID3 algorithm, 这里的树用dict嵌套的方式实现
    Args: Data: 训练集 Label: 这个label指的是每个特征的名字，不是类别label，类别label在Data的最后一列
    returns: decision tree(dict)
    '''
    classlist = [sample[-1] for sample in Data] # 所有可能的类

    # 停止条件1: 如果Data中的样本的类标签都一样
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]

    # 停止条件2: 如果特征被完全用完(只剩类标签)
    if len(Data[0]) == 1:
        return majority(classlist)

    best_feature_index = choose_best_feature(Data, 0)

    # 停止条件3: 信息增益太小
    if best_feature_index == -2:
        return majority(classlist)

    best_feature = Label[best_feature_index]

    myTree = {best_feature: {}} 
        # dict 的 key 是特征名, value_i还是一个dict,新的dict以这轮决策的所有特征值取值作为key,value可能是dict(内结点)，可能是一个classlabel(叶结点)
        # 所以一轮决策其实是有两重dict！
    subLabels = Label[:]
    del(subLabels[best_feature_index])

    featurevalues = [sample[best_feature_index] for sample in Data]
    uniquevalues = set(featurevalues)

    for value in uniquevalues:
        myTree[best_feature][value] = CreateTree(splitData(Data, best_feature_index, value), subLabels)

    return myTree
    

def classify(DecisionTree, Label, testVector):
    '''
    descript: 这个函数对未知的数据testVector进行决策,输出其类别标签
    Args: DecisionTree: 决策树 Label: 特征标签，不是类别标签 testVector: 输入的未知数据的特征向量
    returns: 分类的结果
    '''
    firstStr = list(DecisionTree.keys())[0] # DecisionTree 一定是一个dict,从这个函数最后if-else部分可以看出,firstStr是这轮决策的特征值的名称
    SecondDict = DecisionTree[firstStr] # SecondDict 还是Dict,它的keys是这个特征值的不同取值
    feature_index = Label.index(firstStr)   # 找到这轮决策的特征在Label里的index，才能知道它对应输入特征向量的哪维的值

    key = testVector[feature_index] # 样本的值

    next_node = SecondDict[key]

    if isinstance(next_node, dict):
        outputlabel = classify(next_node, Label, testVector)
    else:
        outputlabel = next_node
    
    return outputlabel


def test():
    '''
    description: 测试函数，计算经验风险
    Args: void
    return: void
    '''
    Data, Label = GetData()
    lenseTree = CreateTree(Data, Label)
    print(lenseTree)
    
    errorCount = 0
    for i in range(len(Data)):
        if classify(lenseTree, Label, Data[i]) != Data[i][-1]:
            errorCount += 1
    print('Error Rate: ',float(errorCount)/len(Data))

if __name__ == "__main__":
    test()
