import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, inv

# 系统参数
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])
Q_value = 1
R_value = 10
Q = Q_value * np.array([[1, 0], [0, 1]])  # 状态代价矩阵
R = R_value * np.array([[1]])           # 控制输入代价矩阵

# LQR 求解
def lqr(A, B, Q, R):
    # 求解离散代数Riccati方程
    P = solve_discrete_are(A, B, Q, R)
    # 计算LQR增益
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K


y0 = random.uniform(-5, 5)
v0 = random.uniform(-5, 5)
y0 = -4.5
v0 = 0
# 初始状态和目标状态
x0 = np.array([[y0], [v0]])  # 初始位置和速度
xd = np.array([[0], [0]])  # 目标位置和速度

# LQR 增益矩阵
K = lqr(A, B, Q, R)

# 仿真参数
T = 10 # 仿真步数
x = x0  # 状态初始化
x_hist = [x.flatten()]  # 保存状态历史
u_hist = []  # 保存输入历史

# 噪声参数
noise_std = 0.1  # 速度噪声的标准差

# 仿真LQR控制
for t in range(T):
    # 计算控制输入
    u = -K @ (x - xd)
    # 生成速度噪声（高斯噪声）
    noise = np.random.normal(0, noise_std, size=(1, 1))
    # 更新状态，加入噪声
    x = A @ x + B @ u + np.array([[0], [noise[0, 0]]])
    # 保存历史数据
    x_hist.append(x.flatten())
    u_hist.append(u.flatten())

# 将状态和输入历史转换为数组
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

# 画图
plt.figure(figsize=(12, 5))

# 位置和速度
plt.subplot(1, 2, 1)
plt.plot(x_hist[:, 0], label='Position y[t]')
plt.plot(x_hist[:, 1], label='Velocity v[t]')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title(f'State of Q={Q_value}, R={R_value}')
plt.xlabel('Time step')
plt.ylabel('State')
plt.legend()

# 控制输入
plt.subplot(1, 2, 2)
plt.plot(u_hist, label='Control input f_i[t]')
plt.axhline(1, color='red', linestyle='--', linewidth=0.8, label='|u| < 1')
plt.axhline(-1, color='red', linestyle='--', linewidth=0.8)
plt.title('Control Input')
plt.xlabel('Time step')
plt.ylabel('u[t]')
plt.legend()

plt.tight_layout()
plt.show()
