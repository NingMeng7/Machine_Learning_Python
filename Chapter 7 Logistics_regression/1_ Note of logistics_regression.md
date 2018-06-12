## Logistics Regression
Logistic regression 是统计学习中的经典分类方法！

## 1.1 Logistic distribution
设X是连续随机变量且服从Logistic distribution,则:
$$F(x) = P(X \leq x) = \frac{1}{1+e^{-\frac{(x-u)}{\gamma}}}$$
$$f(x) = F'(x) = \frac{e^{-\frac{(x-u)}{\gamma}}}{\gamma(1+e^{-\frac{(x-u)}{\gamma}})^2}$$
其中, $\gamma > 0$ 是形状参数, u是位置参数，分布函数F(X)在 $\gamma$ 较小的时候，图像接近单位阶跃函数。

## 1.2 Binomial logistic regression model
$$f:\mathcal{X}\in R \to \mathcal{Y} \in {0,1}$$
其模型由两个条件概率分布给出:
$$P(Y=1|x) = \frac{exp(w·x + b)}{1 + exp(w·x + b)}$$
$$P(Y=0|x) = \frac{1}{1+exp(w·x+b)}$$
对于给定的输入特征向量x,计算两个条件概率分布，将实例x分类到概率更大的类中。

定义: 几率(odds)为事件发生的概率与该事件不发生的概率的比值,即，如果事件发生的概率是p,则其几率为: $\frac{p}{1-p}$,其对数几率为: $log\frac{p}{1-p}$,对logistic regression model而言，有:
$$log \frac{P(Y=1|x)}{1-P(Y=1|x)} = w·x$$  
上式中把偏置项b也放到w中，并用一个分量 1 将特征向量从n dimensions 扩充到 (n+1) dimensions.

可以看出，输出Y=1的对数几率是输入x特征向量的一个线性模型！ 反之，考虑对输入x进行分类的线性函数w·x，其值域为R，考察$P(Y=1|x) = \frac{exp(w·x + b)}{1 + exp(w·x + b)}$, 当线性函数值趋于无穷，概率值接近1，反之，线性函数值趋于负无穷，概率值接近0.

## 1.3 模型参数的估计
给定训练集T, 我们可以应用极大似然法估计模型参数,我们只需要估计一个条件概率,因为两个概率满足和为1，令$\pi(x_i) = P(Y=1|x_i)$有似然函数:
$$\mathcal{L} = \prod_{i=1}^{m} [P(Y=1|x)]^{y_i}[1-P(Y=1|x)]^{1-y_i}$$
$$log \mathcal{L} = \sum_{i=1}^{m} y_i log \frac{\pi(x_i)}{1-\pi(x_i)} + log(1-\pi(x_i)) = \sum_{i=1}^m[ y_i(w·x_i) - log(1+exp(w·x_i)]$$

$$\frac{\partial \mathcal{L}}{w_1} = \sum_{i=1}^m [y_i x_i^{(1)} - \frac{x_i^{(1)}exp(w·x_i)}{1+exp(w·x_i)}]$$

所以更新策略:
$$w \leftarrow  w + Data^T * (Label - sigmoid(Data * w))$$

$sigmoid(Data*w) = [\frac{e(x_i·w)}{1+e(x_i·w)}]_{(mx1)}$

$Label - sigmoid(Data*w) = [y_i - \frac{e(x_i·w)}{1+e(x_i·w)}]_{(mx1)}$

$Data^T * (Label - sigmoid(Data*w)) = [\sum_{i=1}^m x_i^{(j)}(y_i - \frac{e(x_i·w)}{1+e(x_i·w)})]_{(nx1)}$
