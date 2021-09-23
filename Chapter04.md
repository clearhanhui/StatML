[TOC]
# 4.1 朴素贝叶斯法算法
朴素贝叶斯是基于贝叶斯定理和特征条件独立假设的分类方法。朴素贝叶斯法实际上学习到生成数据的机制，所以属于生成模型。条件独立假设等于是说用于分类的特征在类确定的条件下都是条件独立的。这一假设使朴素贝叶斯法变得简单，但有时会牺牲一定的分类准确率。
## 4.1.1 基本方法
设输出空间$\mathcal X \subseteq R^n$为$n$维向量的集合，输出空间为类标记集合$\mathcal Y=\{c_1,c_2,...,c_K\}$。输入为特征向量$X\in \mathcal X$，输出为类标记$Y\in \mathcal Y$。$X$是定义在输入空间$\mathcal X$上的随机向量，$Y$是定义在输出空间$\mathcal Y$。$P(X,Y)$是$X$和$Y$的联合概率分布。训练数据集
$$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$$
**贝叶斯定理**：直接上公式
$$P(Y|X)=\frac{P(X,Y)}{P(X)}=\frac{P(X|Y)P(Y)}{P(X)} \tag{4.1}$$
上式中
$$P(Y=c_k),k=1,2,...,K \tag{4.2}$$
称为先验概率分布；
$$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},X^{(2)}=x^{(2)},...,X^{(n)}=x^{(n)}|Y=c_k),\\
k=1,2,...,K \tag{4.3}$$
称为条件分布概率。
假设$X^{(j)}$可能的取值有$S_j(j=1,2,...,n)$个，那么参数的个数是$K\prod^nS_j$个，是一个指数量级的参数，于是有了下面的假设。
**条件独立性假设**：认为输入变量之间是相互独立的,即
$$\begin{aligned} P(X=x|Y=c_k)
&=P(X^{(1)}=x^{(1)},X^{(2)}=x^{(2)},...,X^{(n)}=x^{(n)}|Y=c_k)\\ 
&=\prod^n_{j=1} P(X^{(j)}=x^{(j)}|Y=c_k),k=1,2,...,K 
\end{aligned}\tag{4.4}$$
将式4.3带入式4.4可以得到朴素贝叶斯的基本公式：
$$P(Y=c_k|X=x)=\frac{P(Y=c_k)\prod_j{P(X^{(j)}=x^{(j)}|Y=c_k)}}{P(X)},\\
k=1,2,...,K\tag{4.5}$$
由此可以得到朴素贝叶斯的分类器：
$$\begin{aligned}y
&=arg\ \underset{c_k}{max}\frac{P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}|Y=c_k)}{P(X)}\\ 
&=arg\ \underset{c_k}{max}P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}|Y=c_k)
\end{aligned}\tag{4.6}$$
## 4.1.2 最大后验的解释
朴素贝叶斯法将实例分到后验概率最大的类中，等价于期望风险最小化。设损失函数为0-1损失函数：
$$L(Y,f(X))=\begin{cases} 
1,\ Y \neq f(X) \\
0,\ Y=f(X)
\end{cases} \tag{4.7}$$
上式中，$f(X)$是分类决策函数。这时期望风险函数为：
$$R_{exp}(f)=E[L(Y,f(X)]=E_X\sum^K_{k=1}[L(c_k,f(X))P(c_k)|X)\tag{4.8}$$
为了使期望风险最小化，只需：
$$\begin{aligned} f(x) 
&=\argmin_{y\in \mathcal Y}\sum^K_{k=1}L(c_k,y)P(c_k|X=x)\\ 
&=\argmin_{y\in \mathcal Y}\sum^K_{k=1}P(y\neq c_k|X=x)\\ 
&=\argmin_{y\in \mathcal Y}\sum^K_{k=1}(1-P(y=c_k|X=x)P(X=x))\\
&=\argmin_{y\in \mathcal Y}\sum^K_{k=1}P(y=c_k|X=x)
\end{aligned}\tag{4.9}$$
# 4.2 朴素贝叶斯法的参数估计
## 4.2.1 极大似然估计
在朴素贝叶斯法中，可以采用就极大似然估计法对$P(Y=c_k)$和$P(X^{(j)}=x^{(j)}|Y=c_k)$进行估计：
$$P(Y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)}{N},k=1,2,...,K\tag{4.10}$$
$$P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^{N}I(x^{(j)}=a_{jl},y_i=c_k)}{\sum_{i=1}^NI(y_i=c_k)}\\
j=1,2,...,n;\ l=1,2,...,S_j;\ k=1,2,...,K\tag{4.11}$$
上式中$x^{(j)}$的取值可能的集合是$\{a_{j1},a_{j2},...,a_{jS_j}\}$；$x_i^{(j)}$是第$i$个样本的第$j$个特征；$a_{jl}$是第$j$个特征可能取的第$l$个值；$I$为指示函数。
## 4.2.2 贝叶斯估计
极大似然估计可能会出现要估计的概率为零的情况，影响计算结果。先验概率和条件概率的贝叶斯估计分别是：
$$P(Y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)+\lambda}{N+\lambda K},\tag{4.12}$$
$$P_{\lambda}(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^{N}I(x^{(j)}=a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^NI(y_i=c_k)+\lambda S_j},\\ 
k=1,2,...,K\tag{4.13}$$
上式中，$\lambda \geq 0$。当$\lambda=0$时，称为拉普拉斯平滑。