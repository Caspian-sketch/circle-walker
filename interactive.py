import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 固定参数
S = np.array([0.0, 0.0])   # 起点
E = np.array([6.0, 0.0])   # 终点
O = (S + E) / 2            # 绿色圆圆心
R = 3.0                    # 绿色圆半径

# 创建图形
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(-4, 10)
ax.set_ylim(-4, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Circle Walker - 拖动滑块旋转虚轴")

# 固定元素：起点、终点、绿色圆
ax.plot(S[0], S[1], 'bo', markersize=10, label='起点')
ax.plot(E[0], E[1], 'ro', markersize=10, label='终点')

theta_circle = np.linspace(0, 2*np.pi, 100)
circle_x = O[0] + R * np.cos(theta_circle)
circle_y = O[1] + R * np.sin(theta_circle)
ax.plot(circle_x, circle_y, 'g--', linewidth=2, label='绿色圆')

# 动态元素（后面会更新）
virtual_axis_line, = ax.plot([], [], 'purple', linewidth=1.5, label='虚轴')
m_point, = ax.plot([], [], 'go', markersize=10, label='M点')
path_line, = ax.plot([], [], 'r--', linewidth=1.5, label='路径')

ax.legend(loc='upper right')

# 创建滑块
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
slider = Slider(ax_slider, '虚轴角度 (度)', 0, 360, valinit=0, valstep=1)

# 更新函数
def update(angle_deg):
    angle_rad = np.radians(angle_deg)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    M = O + R * direction
    
    # 虚轴（从圆心向外延伸）
    axis_end = O + 4 * direction
    virtual_axis_line.set_data([O[0], axis_end[0]], [O[1], axis_end[1]])
    
    m_point.set_data([M[0]], [M[1]])
    path_line.set_data([S[0], M[0], E[0]], [S[1], M[1], E[1]])
    
    ax.set_title(f'虚轴角度: {angle_deg:.0f}°  |  M点: ({M[0]:.2f}, {M[1]:.2f})')
    
    return virtual_axis_line, m_point, path_line

slider.on_changed(update)
update(0)

plt.show()
