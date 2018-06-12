# _*_ coding: utf-8 _*_ 
import numpy as np


Data= np.array([[1,0,3],
                [1,1,3],
                [1,3,2],
                [1,4,4],
               ])
y = np.array([10,12,13,21])


def SGD(Data, Y, eps, rate):
    '''
    description: 这个函数实现代价函数在数据集Data上的随机梯度下降
    Args: Data,Y: training set 注意Data中每个x默认已经加上了x_0 = 1
          threshold: 更新精度
          rate: 学习速率
    returns: weights: 学习到的参数
    '''
    m = Data.shape[0]
    n = Data.shape[1]
    weights = np.zeros(n)   # n x 1
    count = 0

    temp = y - (np.tile(weights,(m,1)) * Data).sum(axis = 1)    # 
    error0 = 0.5 * (temp**2).sum() 
    error1 = 0
    
    while True:
        count += 1

        # 更新部分算法不同
        for i in range(m):
            temp = y[i] - (weights * Data[i]).sum()    
            weights = weights + rate * temp * Data[i]
        
        temp = y - (np.tile(weights,(m,1)) * Data).sum(axis = 1)
        error1 = 0.5 * (temp**2).sum()
        
        if abs(error1 - error0) < eps:
            break
        else:
            error0 = error1
        
        print("Iterations: ", count, "  error: ", error0)    
    return weights

print('SGD',SGD(Data, y, 1e-9,0.03))
