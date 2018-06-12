## 0. Introduction
1. 在有限假设空间上，证明泛化误差可以由训练误差(测试集上的经验误差)和一个和数据集容量N以及假设空间容量M有关的量来约束住(upper bound).
2. 在d->inf,我们提出$m_{\mathcal{H}}$(N) : 有效函数种类, 并提出break point 和 shatter 的概念，证明如果这个模型存在一个break point k，那么我们可以证明，它的 $m_{\mathcal{H}}$(N) 是被一个N的多项式约束住的。
3. 证明，我们可以用这个多项式来替代掉hoeffding's inequality 里的M,在N贡献指数下降的情况下，和一个多项式相乘，证明在这些条件下,训练(测试)误差小的模型，其泛化误差也会小(和第一个相比主要是，拓展到了M infinite的情况),从而证明了，这个模型对未知数据也会有很好的表现，得到learning is possible 的条件。

时刻记得，我们要的是两件事,期望风险(unknown)和经验风险差的不太大，以及经验风险小。

## 1. 泛化误差
假设学习到模型f，则f对 **未知数据** 预测的误差(泛化误差)为:
$$R_{exp}(f) = E_p[L(Y,f(X))] = \int L(y, f(x))P(x,y)dxdy$$
这里的x是未知变量,L是损失函数(风险函数),即,泛化误差是模型f对未知数据的预测期望损失。

## 2.1 有限假设空间的泛化误差的概率上界

泛化误差概率上界，简称为泛化误差上界，具有如下的性质:

1. 它是样本容量N的函数，当样本容量增加，泛化上界趋于0.
2. 它是假设空间容量的函数，假设空间容量越大，模型就越难学，泛化误差上界越大

考虑这样一个训练集 $T = \{(x_1,y_1), (x_2, y_2),..., (x_N, y_N)\}$, 其中各个样本是根据联合概率分布P(X,Y)独立同分布产生的， $X \in R^n, Y \in \{-1, +1\}$ . 假设空间是一个有限的集合$\{f_1,f_2,...,f_d\}$, 设f是从假设空间选取的函数，损失函数采用0-1损失则：
$$R_{exp} (f) = E[L(Y, f(X))]$$
$$R_{emp}(f) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i))$$

在经验误差最小化的策略下选出的函数是: 
$$f_N = arg min R(f)$$
*Theorem 1.1*  泛化误差上界: 对于二分类问题，当假设空间容量等于d有限，对任意一个属于假设空间的函数f，至少以概率$(1-\delta)$ 使得以下不等式成立:
$$R_{exp}(f) \leq R_{emp}(f) + \varepsilon(d, N, \delta)$$
$$\varepsilon(d, N, \delta) = \sqrt{\frac{1}{2N}(log(d)+log(\frac{1}{\delta})}$$

即, 泛化误差可以由测试误差加上一个与假设空间容量，样本容量，和所要求的概率精度有关的修正项来给出一个约束上界.

*Proof:* 

*Lemma:*  设 $S_n = \sum_{i=1}^{n}X_i$ 是独立随机变量 $X_1,X_2,...,X_n$之和, $X_i \in [a_i,b_i]$ 则,对 任意 t>0, 有:
$$P(S_n - E[S_n] \geq t) \leq exp(\frac{-2t^2}{\sum_{i=1}^{n}(b_i - a_i)^2})$$
$$P(E[S_n] - S_n \geq t) \leq exp(\frac{-2t^2}{\sum_{i=1}^{n}(b_i - a_i)^2})$$

考虑任意一个假设空间中的函数f, $R_{emp}$是N个独立随机变量$L(Y, f(X))$ 的样本均值, $R_{exp}(f) = E[R_{emp}(f)]$, 注意样本均值这里分母带着一个N,由于我们取的是0-1损失函数，所以b=1, a=0，进一步有:
$$P(R_{exp}(f)-R_{emp}(f) \geq \varepsilon) \leq exp(-2N\varepsilon^2)$$
再一次,这里右边的N来源于实际上t应该是$N \varepsilon$. 
由于假设空间是有限集,我们有:
$$P(\exists f \in \mathcal{H}: R_{exp}-R_{emp} \geq \varepsilon) $$
$$ = P(\sum_{f \in \mathcal{H}} \{R_{exp}(f) - R_{emp}(f) \geq \varepsilon\})$$
$$\leq \sum_{f \in \mathcal{H}} P(R_{exp}(f) - R_{emp}(f) \geq \varepsilon)$$
$$\leq dexp(-2N\varepsilon^2)$$
在有限集上,存在性等于事件的并集，事件和的概率小于等于事件概率的和，最终利用上面得到的结论对所有f成立即有最后一个不等号。

存在一个f不满足我们结果的概率已经得到，等价地可以得到,对任意假设空间的f，都满足我们结论的概率为:
$$P(R_{exp}(f) - R_{emp}(f) < \varepsilon) \geq 1- dexp(-2N\varepsilon^2)$$
令$\delta =  dexp(-2N\varepsilon^2)$
$$P(R_{exp}(f) - R_{emp}(f) < \varepsilon) \geq 1- \delta$$
*end proof*



可见，测试误差小的模型，泛化误差小也会小, 算法A能依赖于我们的测试集自由地选择h的一个重要的基础就是，由这个不等式给出了泛化误差和训练误差的差距是可以约束的(|$\mathcal{H}$| is limited).

注意经验误差指的是观察值在训练集上的误差，是对期望误差的一个样本估计，而泛化误差是对未知数据的期望误差，测试误差是在测试数据集上对泛化误差的一个样本估计,从概率意义上期望风险和泛化误差是一致的，差别在于针对的对象。

## 2.2 How about d is not finite?
Looking back to our hopes for the conditions:

这里插一句，林轩田老师的课程中，所谓的 bad sample 指的是让经验风险和泛化风险不一致的样本。这份笔记参考的教材比较杂。

1. $R_{exp}$ 和 $R_{emp}$ 差地不太多,这样我们才可以用经验风险来对泛化风险进行估计
2. $R_{emp}$足够小：我们希望找到的g和target f足够接近，能够给出足够好的预测结果。

> 讨论

1. 为了达到条件1，我们显然希望N足够大，同时假设空间的容量d要小一些，在1.5.2中，显然当d是有限的时候，我们学习是可能的。

2. 为了达到条件2,我们显然希望假设空间的容量大一些，算法的选择余地大一些。

如果用1.5.2中的上界估计，如果d趋于无穷的时候，学习似乎不再可能. 进一步考察我们的推导过程，

$$P(\sum_{f \in \mathcal{H}} \{R_{exp}(f) - R_{emp}(f) \geq \varepsilon\})\leq \sum_{f \in \mathcal{H}} P(R_{exp}(f) - R_{emp}(f) \geq \varepsilon)$$
我们将不同的g认为完全没有重叠，但是intuition告诉我们，perceptron模型中，两条非常接近的线，其碰上bad sample 的重叠部分应该非常大才对，因此我们的估计实际上是非常粗糙的。

## 2.3 Effective Number of Lines
以 perceptron 模型为例，输入样本容量N一定的情况下，有很多直线模型是相似的，例如，N=1，实际上只有将这个样本分为正类或者负类两种有效直线，进一步，有；

        N              Max effective(N)
        1                   2
        2                   4
        3                   8
        4                   14 < 2^N

1. must be $\leq 2^N$(since every single sample have two types)
2. finite 'grouping' of infinitely-many lines $\in \mathcal{H}$
3. wish: d can be replaced by effective(N)
4. effective(N) << $2^N$
5. if so, learning is possible even with infinite lines.

## 2.4 Dichotomy: Mini-hypotheses
- call 
$$h(x_1,x_2,...,x_N) = (h(x_1),h(x_2),...,h(x_N)) = \{x,o\}^N$$
a dichotomy: hypotheses 'limited' to the eyes of $x_1, x_2,...,x_N$

Dichotomy: 直线的种类，并且限定在这N个输入样本上来取值.

- $\mathcal{H}(x_1,x_2,...,x_N)$:a set of all the dichotomies 'implemented' by $\mathcal{H} on x_1, x_2,...,x_N$ 


Name | Hypothesis$\mathcal{H}$ | Dichotomies $\mathcal{H}(x_1,x_2,...,x_N)$
-| :-: | -:
e.g. | all lines in $R^2$ | $\{oooo,ooox,ooxx,...\}$
size | possibly infinite | upper bounded by $2^N$

让我们勇敢地做出最坏的估计！我们取的这N个样本中奖了！
$$m_{\mathcal{H}} = max_{x_1,x_2,...,x_N \in \mathcal{X}} |\mathcal{H}(x_1,x_2,...,x_N)|$$ 
(Growth function: bounded by $2^N$)

Before calculating the growth function of perceptron, we may take a glimpse of some simple examples:

> 1. Growth Function for Positive Rays
- $\mathcal{X} = R(dimension=1)$
- $\mathcal{H}$ contains h, where h(x) = sign(x-a) for threshold a
- 门槛函数

N个不同的点，含有N+1种dichotomies. To our 'surprise', (N+1) << $2^N$.

> 2. Growth Function for Positive Intervals
- $\mathcal{X} = R(dimension=1)$
-  $\mathcal{H}$ contains h, where h(x) = +1 iff(if and only if) $x \in [l,r)$, -1 otherwise.

$m_{\mathcal{H}} = C_{N+1}^2 + 1 = \frac{1}{2} N^2 + \frac{1}{2}N + 1 << 2^N$

> 3. Growth Function for Convex Sets 
- $\mathcal{X} = R^2(dimensions=2)$
- $\mathcal{H}$ contains h, where h(x) = +1 iff $x \in convex region$, -1 otherwise.

考虑所有N个样本分布在一个圆上，考虑 Dichotomies: 把想要包括点作为圆的内接多边形顶点，再稍微往外扩一点点覆盖它，这样，所有的可能分类都能实现:

$$m_{\mathcal{H}} = 2^N$$


exists N inputs that can be shattered

## 2.5 What if m replaces d? (Can we? proof later)
$$P[|R_{exp}-R_{emp}| \geq \epsilon] \leq 2m_{\mathcal{H}} exp(-2N\varepsilon^2)$$

If $m_{\mathcal{H}}$ is polynomial:nice! exponential: bad!

下一步,我们就要来对我们想要证明 learning is possible 的模型的growth function 的上界进行估计，请注意，我们不一定要能把 growth function 的具体形式得到，只需要得到上界!

What do we know about 2D perceptrons now?

- 3 inputs: 'exists' shatter!

- 4 inputs: no shatter!

If no k inputs can be shattered by $\mathcal{H}$, call k a **break point** for ${\mathcal{H}}$.

- $m_{\mathcal{H}} < 2^k$ 
- k+1,k+2,k+3,... also break points!

More examples:

1. Positive rays: $m_{\mathcal{H}}$(N) = N + 1 = O(N) break point at 2
2. positive intervals: $m_{\mathcal{H}}$ = O($N^2$) break point at 3
3. convex sets: no break point

## 2.6 B(N,k)
**If no break pont:** $m_{\mathcal{H}} =  2^N!$

**Beak point k**: $m_{\mathcal{H}} = O(N^{k-1})$

> 1. What must be true when minimum break point k = 2

- N = 1 : exist choices of 1 sample: 
$m_{\mathcal{H}} = 2$ (can be shattered)
- N = 2 : for any choices of 2 samples: $m_{\mathcal{H}} < 4$

这部分推导有些复杂，也比较要理解，还是用中文好了->__->。

首先，有效函数种类dichotomies是我们针对一个确定的N个输入样本集，对它们进行分类可能出现的情况，例如，N=2, (x,x) (x,o) (o,x) (o,o)四种情况，这是一个指数的情形，通过举例，我们发现很多情况是可以做到<<于2^N的.

其次,我们用了shatter这个词来代表，dichotomies组成的集合能够产生所有可能的N个数据集的分类情况，也就是2^N,注意，尽管这个集合的容量依赖于这N的数据集的特征，我们这里做最坏估计，只需要存在性即可。

然后是break point，假如对N=k，不管如何选取N个样本点，都无法shatter它们，我们称k是一个break point.

回到这部分，我们假设break point k = 2, 那么任意两个数据点都不能b被shatter! 请注意，我们这里是假设了k = 2，是作为一个希望的条件在使用，我们来看看这会导致什么样的结果。

condition: K = 2

N | max$m_{\mathcal{H}}$ 
-| :-: | -:
1 | 2 
2 | 2/3(since can not be shattered)
3 | 3/4(如果是5,一定会有两个点被shattered)

再一次，shattered: 所有的组合被包括在Dichotomies中. It seems that break point k restricts maximum possible $m_{\mathcal{H}}(N)$ a lot for N > k.

**idea: $m_{\mathcal{H}} \leq$ maximum possioble $m_{\mathcal{H}}(N)$ given k $\leq$ poly(N)**

> 2. Bounding Function(N,K): maximum possible $m_{\mathcal{H}}$(N) when break point = k

B(N,K)是在break point = k的条件下，所有模型的 $m_{\mathcal{H}}$ 容量的一个上界！!我们想要对所有的模型的一个上界做估计。

- combinatorial quantity：

maximum number of length-N vectors with(o,x) while 'no shatter' any length-k subvector. 

- (Why? 让我们抛开具体地区间/门槛/perceptron模型，而转入到一个抽象地B(N,K)函数地研究上) irrelevant of the details of ${\mathcal{H}}$, e.g. B(N,3) bounds both
1. positive intervals (k=3)
2. 1D perceptrons (k=3)

**goal: $m_{\mathcal{H}} \leq$  B(N,k) given k $\leq$ poly(N)**

1. B(N,k) = $2^N$ for N < k
2. B(N,k) = $2^N - 1$ when N = k 上界!

我们现在来证明一个结论:
$$B(N,k) \leq B(N-1,k) + B(N-1,k-1)$$

考虑B(4,3)=11(程序遍历求得),我们把它分成两个部分= 2a + b, 其中2a部分是,(x1,x2,x3)都相同,x4不同的配对部分,b部分是(x1,x2,x3)不配对的部分.

$$B(N,k) = 2a + b$$

1. 去掉$x_4$,还剩下(a+b)个inputs在$(x_1,x_2,x_3)$上，因为任何3个点不能shatter，在这三个样本上显然也不能shatter，所以:

$$a+b \leq B(3,3)$$

2. 去掉$x_4$,去掉不配对地部分，还剩下a个，在这a个样本上不能shatter 2个，否则加上x4(x4是两两配对相反的),就shatter了3个！和B(4,3) k = 3 矛盾,从而:

$$a \leq B(3,2)$$

最后有:
$$B(4,3) = 2a+b \leq B(3,3) + B(3,2)$$

类似上面的证明过程,有:

$$B(N,k) \leq B(N-1,k) + B(N-1,k-1)$$

B(N,k) | K  | 1 | 2 | 3 | 4 | 5 | 6|
-| :-: | :-: | :-: | :-:| :-:| :-:| :-:|
N      | 1  | 1 | 2 | 2 | 2 | 2 | 2
N      | 2  | 1 | 3 | 4 | 4 | 4 | 4
N      | 3  | 1 | 4 | 7 | 8 | 8 | 8
N      | 4  | 1 | <=5  | <=11   | 15| 16| 16
N      | 5  | 1 |  <=6 | <=16  | <=26  | 31| 32
N      | 6  | 1 | <=7 | <=22 | <=43 | <=57

最后，我们转到这样一个结论的证明:

$$B(N,k) \leq \sum_{i=0}^{k-1} C_N^i$$

- k=1,$B(N,1) = 1 \leq 1$ 成立
- 假设k=N时成立.
- $B(N+1,k) \leq B(N,k) + B(N,k-1) \leq \sum_{i=0}^{k-1} C_N^i + \sum_{i=0}^{k-2} C_N^i$
$$= \sum_{i=1}^{k-1} C_N^i + \sum_{i=1}^{k-1} C_N^{i-1} + 1 = \sum_{i=1}^{k-1} C_{N+1}^i + 1$$ 
$$= \sum_{i=0}^{k-1} C_{N+1}^i$$

注意到右边那项的highest term 是$N^{k-1}$，所以，我们得到了:对任意确定的k, B(N,k) upper bounded by poly(N). So,$m_{\mathcal{H}}$(N)is ploy(N) if break point exists!

RK: 事实上，上面那个小于等于号，是等号，去证明大于等于的方向。

因此 2D perceptrons 的成长函数最多是$N^3$阶的,注意B(N,k)给出的是上界，对于一个具体的模型，不一定会到达上界。

## 2.7 我们可以用 m 来替代掉 M吗?
[Vapnik and Chervonenkis, 1971] 这部分证明详见这篇论文
