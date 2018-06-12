## 0. Introduction of Gradient Descent 
Gradient Descent algorithm is one of the most common methods to solve the unconstrained optimization problem. It can be applied easily and we should caculate the gradience of target function at every single loop.
## 1. Description of Gradient Descent algorithm
Suppose that f(x) is a function on $R^n$ and $f(x) \in C^1$. We want to solve the problem：
$$min_{x \in R^n} f(x)$$
The algorithm can be described as follow:
- input: target function *f(x)*, gradient function $g(x) = \nabla f(x)$, computational accuracy $\epsilon$
- output: minimum point of f(x) x* (primarily expected to be global one)
- 1. We start with an initial guess for x, the number of iterations k = 0
- 2. Calculate the value of $f(x^k)$
- 3. update by: $x^{k+1} = x - \alpha g(x^k)$
- 4. if $||f(x^{k+1} - f(x^k)|| < \epsilon$, $x^* = x^{k+1}$. Algorithm stops.
- 5. Otherwise, k = k + 1, go to 2.

## 2. Description of Batch gradient descent algorithm
First of all, let's define the cost function:
$$J(θ) = \frac{1}{2}\sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2 $$
Basically, we want to choose θ so as to minimize J(θ). Note that we have a training set that includes m training samples. We can apply the algorithm in two ways. 

- 1. The algorithm looks at every example in the entire training set on every step.
- 2. The algorithm encounters one example each time and we update the parameters according to the error caused by the single training example.

Let's take a glimpse toward the first method which is called **batch gradient descent**:

Repeat until convergence {
$$θ_j = \theta_j + α \sum^m_{i=1} (y^{(i)} - h_{\theta}(x^{(i)}))x_j^{(i)}  $$
}

Remark: We update every j in the undating step.

## 3. Description of Stochastic gradient descent algorithm
    Repeat until convergence {

        for i = 1 to m: {
            Θj = θj + α(y^i - h(x^i))xj^(i)     for every j
        }
    }

