import numpy as np
from Functions import Create_vocabulary_set
from Functions import textParse
from Functions import text_to_vector


def train_naive_bayes(trainData, trainCategory):
    '''
    descript: 贝叶斯分类器训练
    Args: 训练的文本，每个文本的标签
    returns: P(c), P(xi|c)，由于是二分类，且每个属性是二值的，所以实际返回2个向量和一个P(c)
    '''
    m = len(trainData)
    len_vocab = len(trainData[0])
    pspam = np.sum(trainCategory) / float(m)

    D_spam_attr = np.ones(len_vocab)    #   既是spam,单词又出现的频率统计器
    D_notspam_attr = np.ones(len_vocab)

    D_spam = 2  #   laplace 光滑
    D_notspam = 2

    for i in range(m):
        if trainCategory[i] == 1:
            D_spam_attr += trainData[i]
            D_spam += 1
        else:
            D_notspam_attr += trainData[i]
            D_notspam += 1
    
    pspamVector = (D_spam_attr / D_spam)    #   P(xi|c=spam)
    pnotspamVector = (D_notspam_attr / D_notspam)

    return pspam, pspamVector, pnotspamVector


def classify_naive_bayes(input_vector, pspam, pspamVector, pnotspamVector):
    '''
    description: 对输入的一个代表邮件的向量分类
    Args: 输入邮件的特征向量, 垃圾邮件的先验概率, P(wordi|spam), P(wordi|notspam)
    return: 1(spam) / 0(not spam)
    ''' 
    pnotspam = 1.0 - pspam
    p1 = sum(input_vector * np.log(pspamVector)) + np.log(pspam)	# 乘法是对应元素相乘
    p0 = sum(input_vector * np.log(pnotspamVector)) + np.log(1-pspam)
    input_vector = list(map(lambda x: x*(-1) + 1 , input_vector))
    
    myone = np.ones(len(pspamVector))
    p1 += sum(input_vector * np.log(myone-pspamVector))
    p0 += sum(input_vector * np.log(myone-pnotspamVector))
    if p1 > p0:
    	return 1
    else:
    	return 0


def spamTest():
    '''
    descript: 测试朴素贝叶斯分类器
    Args: void
    return: void
    '''
    docList = []
    classList = []
    fullText = []
    
    for i in range(1, 26):
        wordList = textParse(open(r'C:\Users\xw201\source\repos\6_ Bayes\email/spam/%d.txt' % i, errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(r'C:\Users\xw201\source\repos\6_ Bayes\email/ham/%d.txt' % i, errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    vocabList = Create_vocabulary_set(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
        #print(testSet)
    trainMat = []
    trainClasses = []   
    for docIndex in trainingSet:
        trainMat.append(text_to_vector(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = train_naive_bayes(np.array(trainMat), np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVector = text_to_vector(vocabList, docList[docIndex])
        if classify_naive_bayes(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
        print('The error rate is: ', float(errorCount/len(testSet)))

spamTest()
