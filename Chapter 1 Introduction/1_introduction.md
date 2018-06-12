# Introduction to Statistic Learning
## 1.1 统计学习
统计学习是关于计算机基于数据构建概率统计模型并运用模型对数据进行预测分析的一门学科，也称作统计机器学习。

如果一个系统能够通过执行某个过程改进它的性能,这就是学习。(Acquiring skill with experience accumulated/computed from data).

skill: improve some performance measure(e.g. prediction accuracy)

## 1.1.2 统计学习的对象
统计学习以数据(Data)作为学习对象,从数据出发,提取数据特征，进而抽象出数据模型，再回到对数据的分析和预测。

统计学习关于数据有基本假设: 同类数据(具有某种共同性质的数据)具有一定的统计规律性.

## 1.1.3 统计学习的分类
统计学习的方法是基于数据构建统计模型进而对数据进行预测分系，可以分为

    a) 监督学习(supervised learning)q
    b) 非监督学习(unsupervised learning)    
    c) 半监督学习(semi-supervised learning)
    d) 强化学习(reinforcement learning)

根据I/O变量的连续与离散性质分类

    a) I/O均为连续的称为回归问题
    b) 输出变量为有限个离散变量的预测问题称为分类问题
    c) I/O都是变量序列的预测问题称为标注问题

## 1.1.4 统计学习的方法

考虑监督学习:
    
    a) 对给定的,有限的,用于学习的训练数据(training data)做独立同分布假设
    b) 假设要学习的模型属于某个函数集合: 假设空间(hypothesis)
    c) 应用一定的评价准则(evaluation criterion),从假设空间中选取一个在此准则下对训练数据与未知测试数据最优的模型。
    d) 最优模型的选取通过一定的算法来实现.

**统计学习的三要素**: 模型(model)、策略(strategy)、算法(algorithm)， 我们以策略为准则，通过一定的算法，找到最优模型。

    (1) 

模型可以分为概率模型和非概率模型


## 1.2.0 Hoeffding's Inequality
$$P[|v-u| > \epsilon] \leq 2exp(-2\epsilon^2N)$$
- valid for all N and $\epsilon$
- does not depend on u, **no need to know u**
- larger sample size N or looser gap $\epsilon$ ensure higher probability for v~u

if N is large, v=u probaly approximately correct(PAC)

if large N & i.i.d. $x_n$ can probably infer unknow $[h(x) \neq f(x)]$ probability by known $[h(x_n) \neq y_n$ frction

我们通过取样(**i.i.d基本假设,这个分布无法知道，也不需要知道)**.

for any fixed h, can probably infer $R_{epc}$ by ${R_{emp}}$.

$g ~ f$ is PAC when $R_{emp}$ is small. In other words, if ${R_{emp}}$ is quite big, you can almost sure that $g \neq f$.

RK： **$R_{epc}$ can be far away from ${R_{emp}}$ when involving choice.**

## 1.2.1 经验风险与期望风险

期望风险: $R_{exp}(f) = E_p[L(Y, f(X))] = \int L(y, f(x))P(x,y) dxdy$

经验风险: $R_{emp}(f) = \frac{1}{N}\sum_{i=1}^{N}L(y_i, f(x_i))$

期望风险是无法计算的,而当N趋于无穷，经验风险将会趋于期望风险，但是当样本有限的时候，用经验风险来估计期望风险需要进行矫正: 经验风险最小化 vs 结构风险最小化

## 1.2.2 经验风险最小化与结构风险最小化
经验风险最小化(empirical risk minimization ERM) 策略认为经验风险最小的模型是最好的模型,这是很自然的一个想法，并且当样本比较大的时候效果比较好。

结构风险最小化(structural risk minimization SRM) 是为了防止当样本较小的时候容易出现的过拟合问题，做法是加上一个对模型复杂度的惩罚项,使得结构风险小要求经验风险和模型复杂度同时小(参数lambda大于0)
$$R_{srm}(f) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i)) + \lambda J(f)$$

## 1.3 模型评估与模型选择(策略)

## 1.3.1 训练误差与测试误差

统计学习的目的不仅仅是对已知数据有很好的预测能力，我们还希望它具有相当好的泛化能力，即对未知数据也能给出很好的预测结果。

训练误差: 模型Y=f(x)关于训练数据集的平均损失:
$$R_{emp}(f)=\frac{1}{N}\sum_{i=1}^N L(y_i, f(x_i))$$

测试误差: 模型Y=f(x)关于测试数据集的平均损失:
$$e_{test} = \frac{1}{N'}\sum_{i=1}^{N'}L(y_i,f(x_i))$$

## 1.3.2 过拟合问题

假设空间含有不同复杂度(典型特征是参数的个数)的模型，假设假设空间存在一个真值模型，我们希望我们选择的模型最终很好地逼近这个真值模型，那么我们应该有选择的模型和真模型的参数相同，参数向量与真模型的参数向量接近。

如果只是单纯地追求训练集的效果，可能会导致选择一个复杂度太大的模型，从而导致 over-fitting.

以一般多项式回归问题为例，设M为拟合多项式的阶数，当M增大的时候，测试误差先随着M增大而减小，再随着M增大而增大，后面的增大过程就来源于过拟合的问题。

## 1.4 正则化与交叉验证

## 1.4.1 正则化

模型选择的典型方法是正则化(regularization). 即使得结构风险最小化,在经验风险之后加上一个正则化项(罚项), 正则化项是模型复杂度的单调递增函数。
$$min_f \frac{1}{N}\sum_{i=1}^{N} L(y_i,f(x_i)) + \lambda J(f)$$
例如, 我们可以取定 $\lambda J(f) = ||w||^2$, 从贝叶斯估计的角度来看，正则化的模型选择策略相当于认为简单的模型具有较大的先验概率。

## 1.4.2 交叉验证

如果有充足的样本数据，可以简单地把数据集分为: Training set, validation set(验证集), test set(测试集)分别用来训练模型， 选择模型， 评估学习方法。但是实际情况往往测试数据并不充足，可以采用交叉验证的方法，重复地使用数据，把给定的数据进行切分，再进行组合成为训练集和测试集。

### 1. 简单交叉验证

随机将数据分为两部分,例如0.7的作为训练集，0.3的作为测试集，训练集在不同的情况下训练模型，然后在测试集上评价各个模型的测试误差，选出测试误差最小的模型。

### 2. S折交叉验证

随机地将已给数据切分成S个互不相交地大小相同地子集，利用(S-1)个子集训练模型，然后用剩下地子集测试模型。 将S种可能地划分重复进行，选择S次评测中平均测试误差最小的模型。

### 3. 留一交叉验证

令S = N， 就得到了留一交叉验证，当数据缺乏的时候，我们选择这种S折交叉验证的特殊情况。

## 1.5 泛化能力

泛化能力(generalization ability) 是指由该方法学习到的模型对未知数据的预测能力，是学习方法本质上最重要的能力。 一般来说，我们通过 **测试误差** 来评价学习方法的泛化能力，但是由于样本量有限以及测试样本本身的质量限制，评价结果并不一定完全可靠。

## 1.5.1 泛化误差
假设学习到模型f，则f对 **未知数据** 预测的误差(泛化误差)为:
$$R_{exp}(f) = E_p[L(Y,f(X))] = \int L(y, f(x))P(x,y)dxdy$$
这里的x是未知变量,L是损失函数(风险函数),即,泛化误差是模型f对未知数据的预测期望损失。

## 1.5.2 泛化误差的概率上界

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

## 1.6 生成模型与判别模型

监督学习的任务是学习一个模型,应用这个模型对给定的输入预测相应的输出，其形式一般为
1. 决策函数: Y = f(X)
2. 条件概率分布: P(Y|X)

监督学习方法又可以分为
1. 判别方法(discriminative approach)
2. 生成方法(generative approach)

判别方法是由数据直接学习决策函数$f(X)$ 或者条件概率分布$P(Y|X)$ 作为预测的模型(判别模型，判别方法关心的是给定一个输入X，应该预测什么样的输出Y. 例如:
1. kNN
2. 感知机
3. 决策树
4. 逻辑回归模型
5. 最大熵模型
6. SVM
7. 提升方法
8. 条件随机场 等

生成方法由数据学习联合概率分布$P(X,Y)$, 然后求出条件概率分布$P(Y|X)$ 作为预测模型(生成模型):
$$P(Y|X) = \frac{P(X,Y)}{P(X)}$$
这种方法之所以叫做生成方法，是因为模型表示了给定输出X产生输出Y的生成关系，例如:
1. 朴素贝叶斯法
2. 隐马尔可夫模型

生成方法: 
1. 可以还原出联合概率分布P(X,Y)
2. 生成方法的收敛速度更快，当样本容量N增加的时候，生成方法可以更快地收敛于真实模型
3. 当存在隐变量地时候，仍然可以使用生成方法学习

判别方法:
1. 无法还原P(X,Y), 存在隐变量地时候失效
2. 直接学习的是条件概率P(Y|X)或者决策函数f(X),直接面对预测，学习的准确率比较高
3. 可以对数据进行各种程度上的抽象，定义特征并使用特征，从而简化学习问题

## 1.7 Some Use Scenarios

1. when human cannot program the system manually: navigating on Mars
2. when human cannot define the solution easily: speech/visual recogniton
3. when needing rapid decisions that humans cannot do: high-frequency trading
4. when needing to be user-oriented in a massive scale: consumer-targeted marketing

## 1.8 When to use?
1. exists some 'underlying pattern' to be learned: so performance measure can be improved
2. but no programmable(easy) definition: so ML is needed
3. somehow there is data about the pattern: so ML has some 'input' to learn from

## 1.9 Basic Notations
1. input: $x \in \mathcal{X}$(customer application)
2. output: $y \in \mathcal{Y}$ (good/bad after approving creit card)
3. unknown pattern to be learned(target function) $f : \mathcal{X}  \to \mathcal{Y}$ (ideal credit approval formula)
4. data(training examples): $\mathcal{D} = \{(x_1,y_1), (x_2,y_2),...,(x_N,y_N)\}$ (historical records in bank)
5. hypothesis(skill with hopefully good performance): $g: \mathcal{X} \to \mathcal{Y}$ (learned formula to be used) 

注意, 我们假设f这个真值模型的存在，我们所已知的数据就根据这个f来产生的，但是这个模型是不可知的，否则我们直接根据这个模型编程就可以解决问题也就不需要ML了，我们说g和f很相似，指的是，g和f在相同的输入上给出接近的预测输出。

assume $g \in mathcal{H} = {h_k}$ i.e. approving if:
1. h1: annual salary > NTD 800,000
2. h2: debt > NTD 100,000 (ridiculous but possible)
3. h3: year in job <= 2

hypothesis set $\mathcal{H}$:
1. can contain good or bad hypotheses
2. up to $\mathcal{A}$(learning algorithm) to pick the best one as g 

## 1.10 
1. ML 与 Data Mining share a huge overlap.
2. Statistics can be used for ML
3. ML can realize AI, among other routes
## 习题 1.1
模型: 由参数$\theta$决定的函数族: $P(x=k) =C_k^n \theta^k(1-\theta)^{n-k}$

策略: $\theta = arg max_{\theta} P(Y|X)$ 其中X是已知的观测值

算法: 基于样本的i.i.d假设构造似然函数 $L(\theta) = \prod_{i=1}^{N}P_i(Y_i|X_i,\theta)$, 利用一些数学技巧求出似然函数最大值对应的参数

## 习题 1.2

$$min R_{emp}(f) = min \frac{1}{N}\sum_{i=1}^N L(y_i,f(x_i)) = min \frac{1}{N}\sum_{i=1}^N -lgP(Y_i|X_i)$$
$$max \sum_{i=1}^N lg(Y_i|X_i)$$
显然二者是等价的。
