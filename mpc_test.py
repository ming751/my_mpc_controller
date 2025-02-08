import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 系统参数
J = np.array([[1, 0], [0, 1]])  # 雅可比矩阵
nx = J.shape[0]  # 末端状态维度
nq = J.shape[1]  # 关节角度维度
nu = nq          # 控制输入维度（关节速度）
dt = 0.01         # 时间步长

# MPC 参数
N = 1000 # 预测步长
Q = np.eye(nx)  # 末端状态权重矩阵
R = np.eye(nu)  # 输入权重矩阵
x0 = np.array([0, 0])  # 初始末端状态
q0 = np.array([0, 0])  # 初始关节角度
x_ref = np.array([5, 5])  # 参考末端状态

# 定义优化变量
opti = ca.Opti()

# 状态和输入变量
X = opti.variable(nx, N+1)  # 末端状态轨迹
Q_angle = opti.variable(nq, N+1)  # 关节角度轨迹
U = opti.variable(nu, N)    # 关节速度轨迹

# 初始条件
opti.subject_to(X[:, 0] == x0)
opti.subject_to(Q_angle[:, 0] == q0)

# 动力学约束
for k in range(N):
    opti.subject_to(X[:, k+1] == X[:, k] + J @ U[:, k] * dt)  # 末端状态更新
    opti.subject_to(Q_angle[:, k+1] == Q_angle[:, k] + U[:, k] * dt)  # 关节角度更新

# 输入约束
u_min = -1
u_max = 1
opti.subject_to(opti.bounded(u_min, U, u_max))

# 目标函数
objective = 0
for k in range(N):
    objective += ca.mtimes([(X[:, k] - x_ref).T, Q, (X[:, k] - x_ref)])  # 末端状态误差
    objective += ca.mtimes([U[:, k].T, R, U[:, k]])  # 输入代价

# 终端代价
objective += ca.mtimes([(X[:, N] - x_ref).T, Q, (X[:, N] - x_ref)])

# 最小化目标函数
opti.minimize(objective)

# 求解器设置
opts = {'ipopt.print_level': 0, 'print_time': 0}
opti.solver('ipopt', opts)

# 求解
sol = opti.solve()

# 获取结果
X_opt = sol.value(X)
Q_opt = sol.value(Q_angle)
U_opt = sol.value(U)

# 可视化结果
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(X_opt[0, :], label='x1')
plt.plot(X_opt[1, :], label='x2')
plt.axhline(x_ref[0], color='r', linestyle='--', label='x1_ref')
plt.axhline(x_ref[1], color='g', linestyle='--', label='x2_ref')
plt.legend()
plt.title('End-Effector Trajectory')

plt.subplot(3, 1, 2)
plt.plot(Q_opt[0, :], label='q1')
plt.plot(Q_opt[1, :], label='q2')
plt.legend()
plt.title('Joint Angle Trajectory')

plt.subplot(3, 1, 3)
plt.step(range(N), U_opt.T, label='u')
plt.axhline(u_min, color='r', linestyle='--', label='u_min')
plt.axhline(u_max, color='g', linestyle='--', label='u_max')
plt.legend()
plt.title('Joint Velocity (Control Input)')

plt.tight_layout()
plt.show()