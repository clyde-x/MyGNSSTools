如果按我上面建议的路线（**固定 AB 坐标，用原始双频观测重新估计相对钟差 + 模糊度**），实际上就是一个很标准的线性卡尔曼滤波（甚至不需要 EKF，因为坐标固定后模型基本线性）。

核心思想：

* **钟差：连续变化 → 随机过程（GM/随机游走）**
* **模糊度：弧段内常数 → 常状态**
* **观测：伪距 + 载波**

---

# 1 定义状态量

假设当前历元共有 (m) 颗卫星。

每颗卫星保留一个单差模糊度（双频可以两套）。

状态：

[
x_k=
\begin{bmatrix}
c\delta t_{AB}\
N_{1}^{1}\
N_{2}^{1}\
\vdots\
N_{1}^{m}\
N_{2}^{m}
\end{bmatrix}
]

单位建议统一：

* (c\delta t)：米
* (N)：周

状态维数：

[
n=1+2m
]

---

# 2 状态转移方程

## 钟差：一阶高斯马尔可夫（推荐）

连续形式：

[
\dot x=-\beta x+w
]

离散：

x_k=\Phi x_{k-1}+w_k

其中：

[
\Phi=
\begin{bmatrix}
e^{-\Delta t/\tau} &0\
0&I
\end{bmatrix}
]

即：

[
c\delta t_k
===========

\phi c\delta t_{k-1}
+w_t
]

[
N_k=N_{k-1}
]

其中：

[
\phi=e^{-\Delta t/\tau}
]

经验：

* 原子钟：(\tau\sim1000\sim10000s)
* 普通接收机：(\tau\sim100\sim1000s)

如果不确定：

直接：

[
\phi=1
]

退化成随机游走。

---

过程噪声：

[
Q=
\begin{bmatrix}
\sigma_t^2(1-\phi^2)&0\
0&q_NI
\end{bmatrix}
]

建议：

[
q_N\approx10^{-10}
]

（近似常数）

钟差：

[
\sigma_t\approx0.1\sim10m
]

依设备而定。

---

预测：

[
\hat x_k^-=\Phi \hat x_{k-1}
]

[
P_k^-=\Phi P_{k-1}\Phi^T+Q
]

---

# 3 观测方程

固定：

* 站坐标
* 星历

所以：

[
\rho_{AB}^{(s)}
===============

(\rho_B-\rho_A)
]

已知。

观测减几何：

[
y=z-\rho
]

---

对于卫星 (s)

双频：

伪距：

[
P_1=
c\delta t+I+\varepsilon
]

[
P_2=
c\delta t+\gamma I+\varepsilon
]

载波：

[
L_1=
c\delta t-I+\lambda_1N_1+\varepsilon
]

[
L_2=
c\delta t-\gamma I+\lambda_2N_2+\varepsilon
]

其中：

[
\gamma=\frac{f_1^2}{f_2^2}
]

---

如果不估电离层（短基线）

观测：

[
z=
\begin{bmatrix}
P_1\
P_2\
L_1\
L_2
\end{bmatrix}
]

预测：

[
h(x)=
\begin{bmatrix}
c\delta t\
c\delta t\
c\delta t+\lambda_1N_1\
c\delta t+\lambda_2N_2
\end{bmatrix}
]

于是：

[
H=
\begin{bmatrix}
1&0&0\
1&0&0\
1&\lambda_1&0\
1&0&\lambda_2
\end{bmatrix}
]

---

多卫星时：

直接堆叠。

例如 3 星：

[
z=
[P^1,L^1,P^2,L^2,P^3,L^3]
]

得到稀疏矩阵。

---

# 4 卡尔曼更新

创新：

[
v=z-Hx^-
]

协方差：

[
S=HP^-H^T+R
]

卡尔曼增益：

[
K=P^-H^TS^{-1}
]

更新：

[
x=x^-+Kv
]

[
P=(I-KH)P^-
]

---

# 5 观测噪声怎么设

建议：

伪距：

[
\sigma_P=0.3\sim1m
]

载波：

[
\sigma_L=0.003\sim0.01m
]

因此：

[
R=
diag(
\sigma_P^2,
\sigma_P^2,
\sigma_L^2,
\sigma_L^2
)
]

可以再乘：

[
1/\sin^2(el)
]

做高度角加权。

---

# 6 周跳处理（很重要）

如果某星发生周跳：

删除对应状态：

[
N_i\leftarrow0
]

协方差：

[
P_i\rightarrow10^8
]

重新初始化。

钟差状态连续。

---

这样整个系统就是：

```text
RTKLIB static
     ↓
固定AB坐标

读取RINEX+SP3

构建单差观测

预测:
δt(k)=φδt(k−1)
N(k)=N(k−1)

↓

Kalman更新

↓

输出:
δtAB(t)
```

这个结构本质上已经很接近时间频率领域常见的 **carrier-phase time transfer（CPTT）** 实现了。
