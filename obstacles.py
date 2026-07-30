import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 固定参数
S = np.array([0.0, 0.0])
E = np.array([6.0, 0.0])
O = (S + E) / 2
R = 3.0

# 障碍物（圆心 + 半径）
obs_center = np.array([3.0, 2.5])
obs_radius = 0.8

# 图形
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(-4, 10)
ax.set_ylim(-4, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Circle Walker — 障碍物投影")

# 固定元素
ax.plot(S[0], S[1], 'bo', markersize=10, label='起点')
ax.plot(E[0], E[1], 'ro', markersize=10, label='终点')

# 绿色圆
theta_circle = np.linspace(0, 2*np.pi, 100)
circle_x = O[0] + R * np.cos(theta_circle)
circle_y = O[1] + R * np.sin(theta_circle)
ax.plot(circle_x, circle_y, 'g--', linewidth=2, label='绿色圆')

# 障碍物
obs_circle = plt.Circle(obs_center, obs_radius, color='red', alpha=0.4, label='障碍物')
ax.add_patch(obs_circle)

# 动态元素
virtual_axis_line, = ax.plot([], [], 'purple', linewidth=1.5, label='虚轴')
m_point, = ax.plot([], [], 'go', markersize=10, label='M点')
path_line, = ax.plot([], [], 'r--', linewidth=1.5, label='路径')

ax.legend(loc='upper right')

# 滑块
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
slider = Slider(ax_slider, '虚轴角度 (度)', 0, 360, valinit=0, valstep=1)

# 障碍物投影到绿色圆（计算不可行角度）
def get_forbidden_arc():
    """返回障碍物投影到绿色圆上的不可行角度区间（弧度）"""
    d = np.linalg.norm(obs_center - O)
    if d > R + obs_radius or d < abs(R - obs_radius):
        return []  # 不相交，没有遮挡
    # 计算半角
    alpha = np.arccos((R**2 + d**2 - obs_radius**2) / (2 * R * d))
    beta = np.arctan2(obs_center[1] - O[1], obs_center[0] - O[0])
    start = beta - alpha
    end = beta + alpha
    return [(start, end)]

# 更新函数
def update(angle_deg):
    angle_rad = np.radians(angle_deg)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    M = O + R * direction

    # 检查是否在不可行区间内
    forbidden = get_forbidden_arc()
    if forbidden:
        start, end = forbidden[0]
        # 处理角度回绕
        if start < 0:
            start += 2*np.pi
            end += 2*np.pi
        if angle_rad < 0:
            angle_rad += 2*np.pi
        # 如果在禁区内，推到边界
        if start <= angle_rad <= end:
            # 推到最近的边界
            if angle_rad - start < end - angle_rad:
                angle_rad = start
            else:
                angle_rad = end
            M = O + R * np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # 更新虚轴
    axis_end = O + 4 * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    virtual_axis_line.set_data([O[0], axis_end[0]], [O[1], axis_end[1]])

    m_point.set_data([M[0]], [M[1]])
    path_line.set_data([S[0], M[0], E[0]], [S[1], M[1], E[1]])

    ax.set_title(f'角度: {np.degrees(angle_rad):.0f}°  |  M点: ({M[0]:.2f}, {M[1]:.2f})')

    return virtual_axis_line, m_point, path_line

slider.on_changed(update)
update(0)

plt.show()
