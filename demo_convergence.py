"""
Circle Walker - 多频多绿色圆终点收敛演示
展示波前干涉、绿色圆演化、虚轴选点、事件驱动重建的完整过程
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
import matplotlib.lines as mlines

# ============================================================
# 参数设置
# ============================================================
S_START = np.array([0.0, 0.0])      # 起点
E = np.array([10.0, 0.0])           # 终点
V_WAVE = 1.5                        # 波速
V_AGENT = 0.8                       # S 的移动速度
STEP_SIZE = 0.15                    # 每帧 S 向 M 点移动的距离
BETA = np.radians(45)               # 虚轴最大偏角（弧度）
DT = 0.05                           # 每帧时间步长
EPSILON = 0.3                       # 收敛阈值

# 障碍物列表：每个障碍物 (x, y, radius)
OBSTACLES = [
    (4.0, 1.5, 0.8),
    (6.0, -1.2, 0.7),
    (3.0, -0.8, 0.5),
]

# ============================================================
# 全局状态
# ============================================================
S = S_START.copy()
t_emit = 0.0        # E 发波时刻
t = 0.0             # 全局时间
path = [S.copy()]   # S 的轨迹
rebuild_times = []  # 记录重建时刻
green_circles = []  # 记录绿色圆历史（用于展示多绿色圆）

# ============================================================
# 辅助函数
# ============================================================
def compute_green_circle(S, E, R_S, R_E):
    """计算绿色圆的圆心和半径"""
    d_vec = E - S
    d = np.linalg.norm(d_vec)
    
    if d < 1e-9:
        return None, None
    
    # 检查相交条件
    if R_S + R_E <= d or abs(R_S - R_E) >= d:
        return None, None
    
    # 圆心 O 在线段 SE 上，距 S 的距离为 a
    a = (R_S**2 - R_E**2 + d**2) / (2 * d)
    O = S + (a / d) * d_vec
    
    # 绿色圆半径
    R = np.sqrt(max(R_S**2 - a**2, 0))
    
    return O, R


def project_obstacle(O, R, obs):
    """将障碍物投影到绿色圆上，返回禁入角度区间"""
    ox, oy, r = obs
    obs_center = np.array([ox, oy])
    
    d_vec = obs_center - O
    d = np.linalg.norm(d_vec)
    
    # 障碍物与绿色圆无交集
    if d > R + r:
        return None
    # 障碍物包含圆心
    if d < r:
        return (-np.pi, np.pi)
    
    # 投影半角
    alpha = np.arcsin(np.clip(r / d, -1, 1))
    theta_obs = np.arctan2(d_vec[1], d_vec[0])
    
    return (theta_obs - alpha, theta_obs + alpha)


def select_M(O, R, E, beta, obstacles):
    """虚轴法选择 M 点"""
    # 目标方向
    target_dir = E - O
    theta_target = np.arctan2(target_dir[1], target_dir[0])
    
    # 收集所有禁入区间
    forbidden = []
    for obs in obstacles:
        result = project_obstacle(O, R, obs)
        if result is not None:
            forbidden.append(result)
    
    # 搜索范围
    search_range = np.linspace(theta_target - beta, theta_target + beta, 200)
    
    best_theta = theta_target
    best_dist = float('inf')
    
    for theta in search_range:
        # 检查是否在禁区内
        in_forbidden = False
        for low, high in forbidden:
            # 处理角度环绕
            if low <= theta <= high:
                in_forbidden = True
                break
        
        if not in_forbidden:
            dist = abs(theta - theta_target)
            if dist < best_dist:
                best_dist = dist
                best_theta = theta
    
    # 如果所有角度都被禁用，扩大搜索范围
    if best_dist == float('inf'):
        for theta in np.linspace(theta_target - np.pi/2, theta_target + np.pi/2, 400):
            in_forbidden = False
            for low, high in forbidden:
                if low <= theta <= high:
                    in_forbidden = True
                    break
            if not in_forbidden:
                dist = abs(theta - theta_target)
                if dist < best_dist:
                    best_dist = dist
                    best_theta = theta
    
    M = O + R * np.array([np.cos(best_theta), np.sin(best_theta)])
    return M


# ============================================================
# 绘图设置
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(-2, 12)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Circle Walker - 多绿色圆终点收敛演示')

# 绘制终点
ax.plot(E[0], E[1], 'r*', markersize=20, label='终点 E')
# 绘制障碍物
for obs in OBSTACLES:
    circle = Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.6)
    ax.add_patch(circle)
    ax.plot(obs[0], obs[1], 'x', color='darkred', markersize=8)

# 动态元素
S_point, = ax.plot([], [], 'bo', markersize=10, label='起点 S')
S_wave_circle = Circle((0, 0), 0, fill=False, color='blue', alpha=0.4, linewidth=1.5)
E_wave_circle = Circle((0, 0), 0, fill=False, color='red', alpha=0.4, linewidth=1.5)
green_circle_patch = Circle((0, 0), 0, fill=False, color='green', alpha=0.8, linewidth=2)
ax.add_patch(S_wave_circle)
ax.add_patch(E_wave_circle)
ax.add_patch(green_circle_patch)
# 虚轴和 M 点
virtual_axis, = ax.plot([], [], 'm--', alpha=0.7, linewidth=1.5)
M_point, = ax.plot([], [], 'mo', markersize=8, label='M 点')
# 轨迹
path_line, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1.5, label='S 轨迹')
# 多绿色圆历史记录（半透明）
green_history_patches = []
# 信息文本
info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                     fontfamily='monospace', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
# 图例
ax.legend(loc='upper right')
# ============================================================
# 动画更新函数
# ============================================================
def init():
    S_point.set_data([], [])
    virtual_axis.set_data([], [])
    M_point.set_data([], [])
    path_line.set_data([], [])
    info_text.set_text('')
    return (S_point, virtual_axis, M_point, path_line, info_text,
            S_wave_circle, E_wave_circle, green_circle_patch)
def animate(frame):
    global S, t_emit, t, path, rebuild_times, green_circles
    
    t += DT
    
    # 计算波前半径
    R_S = V_WAVE * t
    R_E = V_WAVE * (t - t_emit)
    
    d_vec = E - S
    d = np.linalg.norm(d_vec)
    
    # 检查收敛
    if d < EPSILON:
        S = E.copy()
        path.append(S.copy())
        update_plot(frame)
        return (S_point, virtual_axis, M_point, path_line, info_text,
                S_wave_circle, E_wave_circle, green_circle_patch)
    
    # 事件驱动：检查是否需要波前重建
    if R_S - R_E >= d and d > 0:
        t_emit = t
        R_E = 0
        rebuild_times.append(t)
    
    # 重新计算 R_E（可能刚重置）
    R_E = V_WAVE * (t - t_emit)
    
    # 计算绿色圆
    O, R = compute_green_circle(S, E, R_S, R_E)
    
    if O is not None and R > 0:
        # 记录绿色圆
        green_circles.append((O.copy(), R, t))
        
        # 虚轴选 M 点
        M = select_M(O, R, E, BETA, OBSTACLES)
        
        # S 向 M 移动
        move_dir = M - S
        move_dist = np.linalg.norm(move_dir)
        if move_dist > 1e-9:
            actual_step = min(STEP_SIZE, move_dist)
            S = S + (move_dir / move_dist) * actual_step
        path.append(S.copy())
    
    # 更新图形
    update_plot(frame)
    
    return (S_point, virtual_axis, M_point, path_line, info_text,
            S_wave_circle, E_wave_circle, green_circle_patch)


def update_plot(frame):
    """更新所有图形元素"""
    R_S = V_WAVE * t
    R_E = V_WAVE * (t - t_emit)
    d = np.linalg.norm(E - S)
    
    # 更新 S 点
    S_point.set_data([S[0]], [S[1]])
    
    # 更新波前圆
    S_wave_circle.set_center((S[0], S[1]))
    S_wave_circle.set_radius(R_S)
    E_wave_circle.set_center((E[0], E[1]))
    E_wave_circle.set_radius(max(R_E, 0))
    
    # 更新绿色圆
    O, R = compute_green_circle(S, E, R_S, R_E)
    if O is not None and R > 0:
        green_circle_patch.set_center((O[0], O[1]))
        green_circle_patch.set_radius(R)
        green_circle_patch.set_visible(True)
        
        # 更新虚轴和 M 点
        M = select_M(O, R, E, BETA, OBSTACLES)
        # 虚轴从 O 到 M 再延伸到圆的另一侧
        dir_vec = M - O
        if np.linalg.norm(dir_vec) > 1e-9:
            dir_unit = dir_vec / np.linalg.norm(dir_vec)
            p1 = O - R * dir_unit
            p2 = O + R * dir_unit
            virtual_axis.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        M_point.set_data([M[0]], [M[1]])
    else:
        green_circle_patch.set_visible(False)
        virtual_axis.set_data([], [])
        M_point.set_data([], [])
    
    # 更新轨迹
    path_arr = np.array(path)
    path_line.set_data(path_arr[:, 0], path_arr[:, 1])
    
    # 更新历史绿色圆（每隔一定帧数添加一个半透明绿色圆）
    if len(green_circles) > 0 and frame % 10 == 0:
        # 清除旧的历史圆
        for patch in green_history_patches:
            patch.remove()
        green_history_patches.clear()
        
        # 显示最近几个绿色圆
        step = max(1, len(green_circles) // 15)
        for i in range(0, len(green_circles), step):
            O_hist, R_hist, t_hist = green_circles[i]
            alpha_val = 0.08 + 0.12 * (i / max(1, len(green_circles)))
            patch = Circle((O_hist[0], O_hist[1]), R_hist, 
                          fill=False, color='green', alpha=alpha_val, linewidth=1)
            ax.add_patch(patch)
            green_history_patches.append(patch)
    
    # 更新信息文本
    info_str = (
        f"时间: {t:.2f}s\n"
        f"t_emit: {t_emit:.2f}s\n"
        f"R_S: {R_S:.2f}  R_E: {R_E:.2f}\n"
        f"S-E距离: {d:.2f}\n"
        f"重建次数: {len(rebuild_times)}\n"
        f"S坐标: ({S[0]:.2f}, {S[1]:.2f})"
    )
    if O is not None:
        info_str += f"\n绿色圆 O: ({O[0]:.2f}, {O[1]:.2f}) R: {R:.2f}"
    info_text.set_text(info_str)
# ============================================================
# 运行动画
# ============================================================
anim = FuncAnimation(fig, animate, init_func=init, frames=None,
                     interval=DT * 1000, blit=False, repeat=False)
plt.tight_layout()
plt.show()
