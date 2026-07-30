import numpy as np
import matplotlib.pyplot as plt

# 参数
S = np.array([0.0, 0.0])   # 起点
E = np.array([6.0, 0.0])   # 终点
O = (S + E) / 2            # 绿色圆圆心
R = 3.0                    # 绿色圆半径

# 生成绿色圆
theta = np.linspace(0, 2*np.pi, 100)
circle_x = O[0] + R * np.cos(theta)
circle_y = O[1] + R * np.sin(theta)

# 选择一个M点（角度45度）
angle = np.pi / 4
M = O + R * np.array([np.cos(angle), np.sin(angle)])

# 绘制
plt.figure(figsize=(6, 6))
plt.plot(circle_x, circle_y, 'g--', label='绿色圆')
plt.plot([S[0], M[0], E[0]], [S[1], M[1], E[1]], 'r--', label='路径')
plt.plot(S[0], S[1], 'bo', label='起点')
plt.plot(E[0], E[1], 'ro', label='终点')
plt.plot(M[0], M[1], 'go', label='M点')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
