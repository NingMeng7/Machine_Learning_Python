import numpy as np
import re

def Create_vocabulary_set(Data):
    '''
    descript: 输入文本构成的单词集合,去除其中重复的部分，得到单词纯集合
    Args: 数据集(含有所有文本内容)
    return: 单词集合
    '''
    vocabulary_set = set()
    for item in Data:
        vocabulary_set = vocabulary_set | set(item)
    return list(vocabulary_set)


def textParse(str):
    '''
    descript: 把文本划分成词
    Args: 文本
    return: 全部都是小写的word列表，长度小于2的被过滤掉
    '''
    token_list = re.split(r'\W+', str)

    return [tok.lower() for tok in token_list if len(tok) > 2]


def text_to_vector(vocabulary_set, input_text):
    '''
    description: 将输入的文本映射到特征空间(向量)
    Args: 单词集合,输入文本
    return: [0,1,0,0,1,...] 其中，0/1 分别代表词典上这个分量代表的单词在输入文本中是否出现(1出现)
    '''
    input_vector = [0] * len(vocabulary_set)  #   创建一个和词典等长的list

    for word in input_text:
        if word in vocabulary_set:
            input_vector[vocabulary_set.index(word)] = 1    # 另一种写法是+=1，但是在计算条件概率的时候要相应做一些变动
        else:
            pass
    
    return input_vector


