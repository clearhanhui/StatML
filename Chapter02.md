[TOC]
# 2.1 感知机
感知机是一种二分类模型，其输入为实例的特征向量，输出为$\pm1$。感知机模型是神经网络和支持向量机的基础。
**定义**：假设输入空间是$X \subseteq R^n$输出空间是$Y \subseteq \{+1,-1\}$。输入$x \in X$表示实例的特征向量，对应于输入空间的点，输出$y \in Y$表示手里的类别，由输入空间到输出空间的如下函数：
$$f(x)=\mathrm{sgn}(w \cdot x + b)$$
称为感知机。其中$w$和$b$成为感知机模型参数，$w \in R^n$称为权值，$b \in R$是偏置，$w \cdot x$表示$w$和$x$的内积，$sgn$是符号函数(或$sign$)，即：
$$\mathrm{sgn}(x) = \begin {cases}  
+1 & x \leq 0 \\
-1 & x < 0
\end{cases}$$
# 2.2 感知机的学习策略
**定义**：给定一个数据集
$$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$$
其中$x_i \in X = R^n, y_i \in Y=\{+1,-1\}, i=1,2,...,N$，如果存在某个超平面$S$
$$w \cdot x+b=0$$
能够将数据集的正实例和负实例点完全正确划分到超平面的两侧，及对所有的$y_i=+1$的实例$i$，有$w\cdot x_i+b >0$，所有的$y_i=-1$的实例$i$，有$w \cdot x_i+b<0$，则称数据集$T$为**线性可分的数据集**；否则，称数据集$T$线性不可分。感知机的学习目标就是寻找一个超平面，能够将一个线性可分数据集完全分离，为了寻找这样的超平面我们需要寻找一个损失函数。这里通计算每个点到超平面的距离总和
$$-\frac{1}{||w||}\sum_{x_i\in M}y_i(w\cdot x_i+b)$$
其中$||w||$是$w$的$L_2$范数，$M$是误分类点集合。忽略$\frac{1}{||w||}$可以得到感知机学习的损失函数(理由参考[这里](https://www.zhihu.com/question/36241719))
$$L(w,b)=-\sum_{x_i\in M}y_i(w\cdot x_i+b)$$
# 2.3 感知机的学习算法步骤
## 2.3.1 算法一
常规算法。首先推导出损失函数$L(w,b)$的梯度是
$$\nabla_wL(w,b)=-\sum_{x_i \in M}y_ix_i$$
$$\nabla_bL(w,b)=-\sum_{x_i \in M}y_i$$
则选取一个误分类点$(x_i, y_i)$，对$w$，$b$进行更新
$$w=w+\eta y_ix_i$$
$$b=b+\eta y_i$$
其中$\eta(0<\eta \leq 1)$是学习率，也称步长。设感知机模型为$f(x)=sgn(w\cdot x+b)$，$\eta(0<\eta \leq 1)$是学习率，也称步长，步骤如下：
1. 选取初值$w_0$，$b_0$；
2. 在训练集中选取数据$\{x_i,y_i\}$；
3. 如果$y_i(w\cdot x_i+b)<0$
$$w=w+\eta y_ix_i$$
$$b=b+\eta y_i$$
4. 转至步骤2，直至没有误分类点，即:
$$L(w,b)=0$$

## 2.3.2 算法一的收敛性
**Novikoff定理**：设训练数据集$T=\{(x_1,y_1),(x_2,y_2),..,(x_N,y_N)\}$是线性可分的，其中$x_i\in X=R^n,y_i\in Y=\{+1,-1\},i=1,2,...,N$，则
1. 存在满足条件$||\hat{w}_{opt}||=1$的超平面$\hat{w}_{opt}\cdot \hat{x}=w_{opt}\cdot x+b_{opt}=0$将训练数据集完全正确分开；且存在$\gamma>0$，对所有$i=1,2,...,N$
$$y_i(\hat{w}_{opt}\cdot \hat{x}_i)=y_i(w_{opt}\cdot x_i+b_{opt})\leq\gamma$$
2. 令$R=max||\hat{x_i}||,1\leq i\leq N$，则感知机算法在训练数据集上的误分类次数$k$满足不等式
$$k\leq(\frac{R}{\gamma})^2$$

证明参考[这里](https://blog.csdn.net/iwangzhengchao/article/details/54486473)
## 2.3.3 算法二
对偶形式。在算法一中，假设初始值$w_0$，$b_0$均为0，对误分类点通过
$$w=w+\eta y_ix_i$$
$$b=b+\eta y_i$$
修改n次，则$w$，$b$关于$(x_i,y_i)$的增量分别是$\alpha_iy_ix_i$和$\alpha_ib_i$，这里$\alpha_i=n_i\eta$，$n_i$表示第$i$个样本被使用的次数。最后学习到的$w$，$b$可以表示为
$$w=\sum_{i=1}^N\alpha_iy_ix_i$$
$$b=\sum_{i=1}^N\alpha_iy_i$$
设感知机模型为$f(x)=sgn(\sum_{j=1}^N\alpha_jy_jx_j \cdot x+b)$，其中$\alpha=\{\alpha_1,\alpha_2,...,\alpha_N\}^T$，步骤如下：
1. $\alpha=0,b=0$；
2. 在训练集中选取数据$(x_i,y_i)$;
3. 如果$y_i(\sum^N_{j=1}a_jy_jx_j\cdot x_i+b)\leq 0$
$$a_i=a_i+\eta$$
$$b=b+\eta y_i$$
4. 转至步骤2，直至没有误分类点，即:
$$L(w,b)=0$$

对偶形式中训练实例仅以内积的形式出现。为了方便，可以预先将训练集中实例间的内积计算出来并以矩阵的形式存储，，这个矩阵就是所谓的**Gram矩阵**。
$$G=[x_i\cdot x_j]_{N \times N}$$
