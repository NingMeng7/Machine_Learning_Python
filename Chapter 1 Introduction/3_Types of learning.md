## 1.1 More Binary Classification Problems
1. credit approve/disapprove
2. email spam/non-spam
3. patient sick/not sick
4. ad profitable/not profitable
5. answer corret/incorrect

## 1.2 Multiclass Classification: Coin Recognition Problem
- classify US coins (1c, 5c, 10c, 25c) by (size, mass)
- y = {1, 2, 3,...,K}
- binary classification:special case with K = 2

1. writtten digits => 0,1,...,9
2. pictures => apple, orange, strawberry
3. emails => spam, primary, social, promotion, updata

recognition!

## 1.3 Regression Recovery Prediction Problem

- Binary classification: patient features => sick or not
- multiclass classification: patient features => which type of cancer
- regression: patient features => how many days before recovery
- y = R or y = [lower, upper] (bounder regression)
- These kinds of problems are deeply studied in statistics

## 1.4 Structured Learning: Sequence Tagging Problem
- multiclass classification: word => word class
- structured learning: sentence => structure(class of each word)
- y = {PVN, PVV, NVN,PV,...}
- huge multiclass classfication problem (structure = hyperclass) without 'explicit' class definition

1. protein data => protein folding
2. speech data => speech parse tree
3. a fancy but complicated learning problem

## 2.1 Supervised: Coin Recognition Revistited

一个简单的判断方法是，如果输入数据有标签(label),就是有监督学习，这代表着我们要解决的问题是有着标准的答案的,我们要做的是从已知的样本中学习特定答案的特点，从而在面对新的未知数据的时候能够给出与其正确答案相符的预测。

**supervised learning: every $x_n$ comes with corresponding $y_n$**

## 2.2 Unsupervised: Coin Recognition without $y_n$

Because we don't give it the answer, it's unsupervised learning.

unsupervised multiclass classification(clustering): a challenging but useful problem

- clustering: $\{x_n\}$ => cluster(x) (almost equal 'unsupervised multiclass classification') i.e. article => topics

- density estimation: $\{x_n\} => density(x)$ (almost equal 'unsupervised bounded regression') i.e. traffic reports with location => dangerous areas
- outlier detection: $\{x_n\} => unusual(x)$ (almost equal 'extreme 'unsupervised binary classification') i.e. Internet logs => intusion alert

**unsupervised learning: diverse, with possibly very different performance goals**

## 2.3 Semi-supervised: Coin Recognition with Some $y_n$

1. face images with a few labeled => face identifier(Facebook)
2. medicine data with a few labeled => medicine effect predictor

Labeling costs a lot!

**Semi-supervised learning: leverage unlabeled data to avoid 'expensive' labeling**

## 3 Reinforcement Learning
A very different but natural way of learning

Teach Your Dog: Say 'Sit Down'

> 1. Maybe 2


The dog pees on the ground

BAD DOG. THAT'S A VERY WARONG ACTION

- cannot easily show the dog that $y_n = sit$ when $x_n = 'sit down'$
- but can 'punish' to say $y'_n$ = pee is wrong

> 2. Maybe 1

The dog sits down.

Good Dog. Let me give you some cookies.

- still cannot show $y_n$ = sit when $x_n$ = 'sit down'
- but can 'reward' to say $y'_n$ = sit is good


(x y', goodness)
1. (customer, ad choice, ad click earning) => ad system
2. (cards, stratery, winning amount) => black jack agent

**Reinforcement: learn with 'partial/implicit information' (often sequentially)**

## 4.1 Batch Learning Problems
**batch supervised multiclass classification: learn from all known data**

1. batch of (email, spam?) => spam filter
2. batch of (patient, cancer) => cancer classifier
3. batch of patient data => group of patients

## 4.2 Online: Spam Filter that 'Improves'
- batc spam filter: learn with known(eamil, spam?) pairs, and predict with fixed g
- online spam filter, which sequentially
1. observe an email x_t
2. predict spam status with current $g_t(x_t)$
3. receive 'desired label' $y_t$ from user, and then update $g_t$ with $(x_t, y_t)$

1. PLA can be easily adapted to online protocol
2. reinforcement learning is often done online

**Online: hypothesis 'improve' through receiving data instances sequentially.**

## 4.3 Active Learning: Learning by 'Asking'
1. batch: 'duck feeding'
2. online:'passive sequential'
3. active:'question asking' (sequentially) - query the $y_n$ of the chosen $x_n$

**Active: improve hypothesis with fewer labels(hopefully) by asking questions strategically**

## 5.1 Features of input

> 1. concrete features : each dimension of $x \in R^d represents 'sophisticated physical meaning'$

1. (size, mass) for coin classification
2. customer info for credit approval
3. patient info for cancer diagnosis
4. often including 'human intelligence' on the learning task

> 2. Raw Features
For instance, digit recognition problem:

16 by 16 gray image x = (0,0,0.9,0.6...)$\in R^{256}$

'simple physical meaning' thus more difficult for ML than concrete features

> 3. Abstract Features: Rating Prediction Problem
Rating prediction problem(KDDCup 2011)

- given previous(userid, itemid, rating) tuples, predict the rating that some userid would give to itemid?
- a regression problem with problem with $y \subset R$ as rating and $x \subset N x N$ as (userid, itemid)
- 'no physical meaning', thus even more difficult for ML

**abstract: again need 'feature conversion/extraction/construction'**


