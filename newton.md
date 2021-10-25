# 牛顿法和拟牛顿法
最早接触牛顿法是本科时候学《数值计算方法》，现在重新捡起来。
### 牛顿法
考虑无约束最优化问题
$$
\begin{equation}
\min \limits _{x \in \mathbb{R} ^ n} f(x)
\end{equation}
$$
其中， $x^*$ 为目标函数的极小点。

假设 $f(x)$ 具有二阶连续偏导数，若第 $k$ 次迭代值为$x^(k)$，则可将 $f(x)$ 在 $x^{(k)}$ 附近进行二阶泰勒展开：
$$
\begin{equation}
f(x) = f(x^{(k)}) + g^T_k(x-x^{(k)})+1/2(x-x^{(k)})^T H(x^{(k)})(x-x^{(k)})
\end{equation}
$$
这里， $g_k = g(x^{(k)})=\nabla f(x^{(k)})$ 是 $f(x)$ 的梯度向量在点 $x^{(x)}$ 的值， $H(x^{(k)})$ 是 $f(x)$ 的黑塞矩阵（hessian matrix）：
$$
\begin{equation}
H(x) = 
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_i \partial x_j}
\end{bmatrix} _{n \times n}
\end{equation}
$$
在点 $x^{(k)}$ 的值。函数 $f(x)$有极值的必要条件是在极值点处一阶导数为 0，即梯度向量为 0。当 $H(x^{(k)})$ 是正定矩阵时，函数 $f(x)$ 的极值为极小值。
牛顿法利用极值点的必要条件：
$$
\begin{equation}
\nabla f(x) = 0
\end{equation}
$$
每次迭代中从 $x^{(k)}$ 开始，求目标函数的极小点，作为 $k+1$ 次迭代值 $x^{(k+1)}$ 。具体地, 假设 $x^{(k+1)}$ 满足:
$$
\begin{equation}
\nabla f(x^{(k+1)})=0
\end{equation}
$$
由式 $(2)$ 有（经过移项变换）
$$
\begin{equation}
\nabla f(x)=g_{k}+H_{k}(x-x^{(k)})
\end{equation}
$$
其中 $H_{k}=H(x^{(k)})$ 。这样, 式 (5) 成为
$$
\begin{equation}
g_{k}+H_{k}(x^{(k+1)}-x^{(k)})=0
\end{equation}
$$
因此,
$$
\begin{equation}
x^{(k+1)}=x^{(k)}-H_{k}^{-1} g_{k}
\end{equation}
$$
或者
$$
\begin{equation}
x^{(k+1)}=x^{(k)}+p_{k}
\end{equation}
$$
其中，
$$
\begin{equation}
H_{k} p_{k}=-g_{k}
\end{equation}
$$
公式 $(8)$ 作为迭代公式即牛顿法。

算法：  
输入：目标函数 $f(x)$, 梯度 $g(x)=\nabla f(x)$, 黑塞矩阵 $H(x)$, 精度要求 $\varepsilon$;   
输出：$f(x)$ 的极小点 $x^{*}$ 。  
* (1) 取初始点 $x^{(0)}$, 置 $k=0$ 。
* (2) 计算 $g_{k}=g(x^{(k)})$ 。
* (3) 若 $||g_{k}||<\varepsilon$, 则停止计算, 得近似解 $x^{*}=x^{(k)}$ 。
* (4) 计算 $H_{k}=H(x^{(k)})$, 并求 $p_{k}$
$$
H_{k} p_{k}=-g_{k}
$$
* (5) 置 $x^{(k+1)}=x^{(k)}+p_{k}$ 。
* (6) 置 $k=k+1$, 转 $(2)$ 。
步骤 (4) 求 $p_{k}, p_{k}=-H_{k}^{-1} g_{k}$, 要求 $H_{k}^{-1}$, 计算比较复杂, 所以有其他改进的方法。

### 拟牛顿法
针对上述步骤（4）中计算复杂度的问题，考虑使用一个n阶矩阵 $G_k=G(x_{(k)})$ 来代替 $H^{-1}_k$ 。
在牛顿 $(6)$ 中取 $x=x^{(k+1)}$, 即得
$$
\begin{equation}
g_{k+1}-g_{k}=H_{k}(x^{(k+1)}-x^{(k)})
\end{equation}
$$
记 $y_{k}=g_{k+1}-g_{k}, \delta_{k}=x^{(k+1)}-x^{(k)}$, 则
$$
\begin{equation}
y_{k}=H_{k} \delta_{k} \ \text{或} \ H_{k}^{-1} y_{k}=\delta_{k}
\end{equation}
$$
称为**拟牛顿条件**。

如果 $H_{k}$ 是正定的 $(H_{k}^{-1}$ 也是正定的)，那么可以保证牛顿法搜索方向 $p_{k}$ 是下降 方向。这是因为搜索方向是 $p_{k}=-H_{k}^{-1} g_{k}$，由式 $(B.8)$ 有
$$
\begin{equation}
x=x^{(k)}+\lambda p_{k}=x^{(k)}-\lambda H_{k}^{-1} g_{k}
\end{equation}
$$
所以 $f(x)$ 在 $x^{(k)}$ 的泰勒展开式 $(B .2)$ 可以近似写成:
$$
\begin{equation}
f(x)=f(x^{(k)})-\lambda g_{k}^{\mathrm{T}} H_{k}^{-1} g_{k}
\end{equation}
$$
因 $H_{k}^{-1}$ 正定, 故有 $g_{k}^{\mathrm{T}} H_{k}^{-1} g_{k}>0$ 。当 $\lambda$ 为一个充分小的正数时, 总有 $f(x)<f(x^{(k)})$, 也就是说 $p_{k}$ 是下降方向。
拟牛顿法将 $G_{k}$ 作为 $H_{k}^{-1}$ 的近似, 要求矩阵 $G_{k}$ 满足同样的条件。首先, 每次迭 代矩阵 $G_{k}$ 是正定的。同时, $G_{k}$ 满足下面的拟牛顿条件:
$$
\begin{equation}
G_{k+1}\ y_{k}=\delta_{k}
\end{equation}
$$

根据拟牛顿条件选择 $G_k$ 作为 $H^{-1}_k$ 的近似，或选择 $B_k$ 作为 $H_k$ 的近似的算法为拟牛顿法。不同的合理的选择有不同的实现方法，接下来只介绍BFGS（Broyden-Fletcher-Goldfarb-Shanno），一个比较流行的算法。

假设每一步的迭代过程 $B_{k+1}$ 是由 $B_k$ 加上两个附加项构成的，即：
$$
\begin{equation}
B_{k+1}=B_k + P_k + Q_k
\end{equation}
$$
且满足拟牛顿条件：
$$
\begin{equation}
B_{k+1} \delta _k =B_k \delta _k + P_k \delta _k + Q_k \delta _k
\end{equation}
$$
为了满足上述拟牛顿条件，可以令：
$$
\begin{equation}
\begin{aligned}
P_{k} \delta_{k}&=y_{k} \\
Q_{k} \delta_{k}&=-B_{k} \delta_{k}
\end{aligned}
\end{equation}
$$
找出适合条件的 $P_{k}$ 和 $Q_{k}$, 得到 BFGS 算法矩阵 $B_{k+1}$ 的迭代公式:
$$
\begin{equation}
B_{k+1}=B_{k}+\frac{y_{k} y_{k}^{\mathrm{T}}}{y_{k}^{\mathrm{T}} \delta_{k}}-\frac{B_{k} \delta_{k} \delta_{k}^{\mathrm{T}} B_{k}}{\delta_{k}^{\mathrm{T}} B_{k} \delta_{k}}
\end{equation}
$$
可以证明, 如果初始矩阵 $B_{0}$ 是正定的, 则迭代过程中的每个矩阵 $B_{k}$ 都是正定的。

算法：  
输入：目标函数 $f(x), g(x)=\nabla f(x)$, 精度要求 $\varepsilon$;  
输出： $f(x)$ 的极小点 $x^{*}$ 。  
* (1) 选定初始点 $x^{(0)}$, 取 $B_{0}$ 为正定对称矩阵, 置 $k=0$ 。
* (2) 计算 $g_{k}=g\left(x^{(k)}\right)$ 。若 $||g_{k}||<\varepsilon$, 则停止计算, 得近似解 $x^{*}=x^{(k)}$; 否则转 (3)。
* (3) 由 $B_{k} p_{k}=-g_{k}$ 求出 $p_{k}$ 。
* (4) 一维捜索:求 $\lambda_{k}$ 使得
$$
f\left(x^{(k)}+\lambda_{k} p_{k}\right)=\min _{\lambda \geqslant 0} f\left(x^{(k)}+\lambda p_{k}\right)
$$
* (5) 置 $x^{(k+1)}=x^{(k)}+\lambda_{k} p_{k}$ 。
* (6) 计算 $g_{k+1}=g\left(x^{(k+1)}\right)$, 若 $||g_{k+1}||<\varepsilon$, 则停止计算, 得近似解 $x^{*}=x^{(k+1)}$; 否则, 按式 (B.30) 算出 $B_{k+1}$ 。
* (7) 置 $k=k+1$, 转 (3)。


```python
# newton algorithm written in python
def f(x):
    return math.exp(x) + x**2 + x*3 + 5.0

def df(x):
    return math.exp(x) + 2*x + 3.0

def ddf(x):
    return math.exp(x) + 2.0
    
def newton_optimize(x_init=0, eps=0.01):
    epoch = 1
    x_0 = x_init
    x_1 = x_0 - df(x_0)/ddf(x_0)
    while abs(x_1-x_0) > eps:
        x_0 = x_1
        x_1 = x_0 - df(x_0)/ddf(x_0)
        epoch+=1
    return x_1, epoch

if __name__ == "__main__":
    x_opt, epoch = newton_optimize(0, 1e-4)
    print(epoch, x_opt, f(x_opt))

```