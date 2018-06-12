## 1. array.argsort()
array.argsort()
返回的是数组值升序排列的索引值
例如: 
>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])

值最小的索引值为1，最大的索引值为0



编程问题：

1. 归一化数据的代码中，返回的数组返回成了原数组，这个bug找了1个半小时。
