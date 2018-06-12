## Perceptron

## 2.1 Description of problem
For $x = (x_1, x_2,...,x_n)$(features of custome), compute a weighted 'score' and 
1. approve credit if $\sum_{i=1}^{n} w_ix_i > threshold$
2. deny credit if $\sum_{i=1}^{n} w_ix_i < threshold$

$y:\{+1(good), -1(bad)\}$ - linear formula $h \in \mathcal{H}$ are :
$$h(x) = sign((\sum_{i=1}^{n} w_ix_i - threshold))$$

或者把threshold也放进向量中:
$$h(x) = sign(\sum_{i=0}^{n} w_ix_i) = sign(w^T·x)$$

## 2.2 Possible Hypothesis Set
在感知机的例子中, 我们的$\mathcal{H}$是一个由参数向量$w^T$ 决定的线性分类器,这是我们所确定的假设空间集合。

## 2.3.1 从假设空间中找出g
1. want: g~f (hard since f is unknown)
2. Naturally: almost necessary: g~f on $\mathcal{D}$, ideally, $g(x_n) = f(x_n) = y_n$
3. difficult: $\mathcal{H}$ is infinite!!
4. idea: start from some $g_0$, and 'correct' its mistakes on $\mathcal{D}$. We represent g with $w^T$


## 2.3.2 具体的做法
1. find a mistake of $w_t$ called $(x_{n(t)}, y_{n(t)}), sign(w^T_t·x_{n(t)} \neq y_{n(t)})$
2. (try to) correct the mistaked by
$$w_{t+1} \leftarrow w_t + y_{n(t)}x_{n(t)}$$ 
3. ...until no more mistakes

## 2.4 PLA 什么时候停下?
If PLA halts(i.e. no more mistakes), (necessary condition) $\mathcal{D}$ allows some w to make no mistake (call such $\mathcal{D}$ Linear separable), then:

$$y_{n(t)}w_f^Tx_{n(t)} \geq min_n y_n w_f^Tx_n > 0$$

by updating with any $(x_{n(t)}, y_{n(t)}$, we get:
$$w_f^Tw_{t+1} = w_f^T(w_t+y_{n(t)}x_{n(t)}) \geq w_f^Tw_t + min_n y_nw_f^Tx_n > w_f^Tw_t + 0$$
$w_t$ appears more aligned w_f after update. However, you can't ignore the influence of ||w||!

$w_t$ changeed only when mistake! i.e. $sign(w_t^Tx_{n(t)} \neq y_{n(t)} <=> y_{n(t)}w_t^Tx_{n(t)} \leq 0$ 
$$||w_{t+1}||^2 = ||w_t + y_{n(t)}x_{n(t)}||^2 = ||w_t||^2 + 2y_{n(t)}w_t^Tx_{n(t)}+||y_{n(t)}x_{n(t)}||^2 \leq ||w_t||^2 + 0 + ||y_{n(t)}x_{n(t)}||^2$$
$$\leq ||w_t||^2 + max_n||y_nx_n||^2$$

即有,当我们只对错误分类的样本进行修正,我们会发现:
$$\frac{w_f^T}{||w_f||} \frac{w^T}{||w_T||} \geq \sqrt{T}·constant$$

Define $R^2 = max_n||x_n||^2$ $p = min_n y_n \frac{w_f^T}{||w_f||} x_n$
$$Since: 1 \geq \frac{w_f^T}{||w_f||} \frac{w^T}{||w_T||} \geq \sqrt{T}·constant$$
We can get a upper bound: 
$$\frac{R^2}{p^2}$$

*Novikoff:* *considering T = $\{(x_1, y_1), (x_2, y_2),...,(x_N, y_N)\}$, and $\mathcal{D}$ is linear separable. Then, we have the conclusions:*

*(i) $\exists w_f$ such that $w_f·x + b_f = 0$ separate the whole $\mathacal{D}$ with no mistake. Besides, $\exists \rho > 0$ such that for all samples, we have:*
$$y_i(w_f·x_i + b_f) \geq \rho$$

*(ii) let $R = max_{\mathcal{D}}||x_i||$, we get an upper bound of the number of mistake classifications: *
$$k \leq (\frac{R}{\rho})^2$$

*proof*

(1) $w$的存在来源于线性可分的基本假设,而$||w_f|| = 1$的保证是，在进行归一化后，并不会影响判断的正负性，也就是说，仍然能保证所有样本被正确分类，注意，只有符号起作用！因此我们有:$||w_f||=1$ 且:
$$y_i(w_f·x_i) > 0$$
由于$\mathcal{D}$是有限的,因此:
$$y_i(w_f·x_i) \geq \rho = min_i\{y_i(w_f·x_i)\}$$
第k个误分类的实例满足:
$$y_i(w_{k-1}·x_i) \leq 0$$
做更新:
$$w_k = w_{k-1} + y_ix_i$$

我们有:
$$w_k·w_f = w_{k-1}·w_f + y_iw_f·x_i \geq w_{k-1}·w_f + \rho \geq w_{k-1}·w_f + 2\rho \geq ... \geq k\rho $$
$$||w_k||^2 = ||w_{k-1}||^2 + 2y_iw_{k-1}·x_i+||x_i||^2 \leq ||w_{k-1}||^2+||x_i||^2 \leq ||w_{k-1}||^2+R^2 \leq ... \leq kR^2$$

注意上面第二个不等式的第一步来源于余弦定理, 最终有:
$$k\rho \leq w_k·w_f \leq ||w_k||||w_f|| \leq \sqrt{k}R$$
$$k \leq (\frac{R}{\rho})^2$$
*end*
## 2.5 Brief Summary
>as long as linear separable and correct by mistake
- inner product of $w_f$ and $w_t$ grows fast; length of $w_t$ grows slowly
- PLA 'lines' are more and more aligned with $w_f$ => halts

> Pros
- simple to implement, fast, works in any dimension d

> Cons
- We must assume linear separable to halt (property unknown in advance(no need for PLA if we know $w_f$))
- not fully sure how long halting use(p depends on $w_f$! which is unknown)

> What if not linear separable?
- assume that the $\mathcal{D}$ includes some noice. We may never find a g!!
- 解决办法,我们不要求g完全分割整个训练集，我们仍然以 g~f 作为要求:
$$w_g \leftarrow arg min_w \sum_{n=1}^{N}[y_n \neq sign(w^Tx_n)]$$

## 2.6 Pocket Algorithm
modify PLA algorithm by keeping best weights in pocket

initialize pocket weights w

For t = 0,1,...
1. find a (random) mistake of $w_t$ called (x_{n(t)}, y_{n(t)})
2. (try to) correct the mistake by
$w_{t+1} \leftarrow w_t + y_{n(t)}x_{n(t)}$ 
3. if $w_{t+1}$ makes fewer mistakes than w, replace w by $w_{t+1}$... until iterations(we cannot ensure that this algorithm will halt like PLA)

Since we do not know whether $\mathcal{D}$ is linear separable in advance, we may decide to just go with the pocket algorithm. If $\mathcal{D}$ is actually linear separable, this algorithm is slower than PLA.
