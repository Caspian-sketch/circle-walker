"""
Circle Walker - M 点路径 + 障碍物避让演示
M1 → M2 → M3 → ... 通过虚轴偏转（β 限制）绕过障碍物，最终收敛到终点
上图：绿色圆波动 + 虚轴避障选 M 点
下图：M 点连线绕过障碍物收敛到终点
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arc
import matplotlib.patches as mpatches

# ============================================================
# 参数设置
# ============================================================
S_START = np.array([-5.0, 0.0])
E = np.array([5.0, 0.0])
V_WAVE = 1.5
DT = 0.04
STEP_SIZE = 0.1
BETA = np.radians(55)
MAX_TIME = 20.0

# 障碍物：堵在 S-E 连线上，留出上下绕行空间
OBSTACLES = [
    (1.0, 0.0, 0.8),
]

# ============================================================
# 全局状态
# ============================================================
S = S_START.copy()
t_emit = 0.0
t = 0.0
path = [S.copy()]
green_circles_history = []
M_points = []
last_theta = None
# ============================================================
# 辅助函数
# ============================================================
def compute_green_circle(S, E, R_S, R_E):
    d_vec = E - S
    d = np.linalg.norm(d_vec)
    if d < 1e-9:
        return None, None
    if R_S + R_E <= d or abs(R_S - R_E) >= d:
        return None, None
    a = (R_S**2 - R_E**2 + d**2) / (2 * d)
    O = S + (a / d) * d_vec
    R = np.sqrt(max(R_S**2 - a**2, 0))
    return O, R


def project_obstacle(O, R, obs):
    ox, oy, r = obs
    obs_center = np.array([ox, oy])
    d_vec = obs_center - O
    d = np.linalg.norm(d_vec)
    if d > R + r:
        return None
    if d < r:
        if d < 1e-9:
            return (-np.pi, np.pi)
        cos_alpha = (d**2 + R**2 - r**2) / (2 * d * R)
        cos_alpha = np.clip(cos_alpha, -1, 1)
        alpha = np.arccos(cos_alpha)
    else:
        alpha = np.arcsin(np.clip(r / d, -1, 1))
    theta_obs = np.arctan2(d_vec[1], d_vec[0])
    low = theta_obs - alpha
    high = theta_obs + alpha
    return (low, high)


def select_M(O, R, E, beta, obstacles, last_theta):
    target_dir = E - O
    theta_target = np.arctan2(target_dir[1], target_dir[0])

    forbidden = []
    for obs in obstacles:
        result = project_obstacle(O, R, obs)
        if result is not None:
            forbidden.append(result)

    search_angles = np.linspace(theta_target - beta, theta_target + beta, 300)

    best_theta = theta_target
    best_dist = float('inf')

    for theta in search_angles:
        in_forbidden = False
        for low, high in forbidden:
            if low <= high:
                if low <= theta <= high:
                    in_forbidden = True
                    break
            else:
                if theta >= low or theta <= high:
                    in_forbidden = True
                    break

        if not in_forbidden:
            dist = abs(theta - theta_target)
            if dist < best_dist:
                best_dist = dist
                best_theta = theta

    if best_dist == float('inf'):
        for theta in np.linspace(theta_target - np.pi/2, theta_target + np.pi/2, 500):
            in_forbidden = False
            for low, high in forbidden:
                if low <= high:
                    if low <= theta <= high:
                        in_forbidden = True
                        break
                else:
                    if theta >= low or theta <= high:
                        in_forbidden = True
                        break
            if not in_forbidden:
                dist = abs(theta - theta_target)
                if dist < best_dist:
                    best_dist = dist
                    best_theta = theta

    if last_theta is not None:
        old_available = True
        for low, high in forbidden:
            if low <= high:
                if low <= last_theta <= high:
                    old_available = False
                    break
            else:
                if last_theta >= low or last_theta <= high:
                    old_available = False
                    break
        if old_available and abs(best_theta - last_theta) < np.radians(5):
            best_theta = last_theta

    M = O + R * np.array([np.cos(best_theta), np.sin(best_theta)])
    return M, best_theta, forbidden
# ============================================================
# 预计算所有帧的数据
# ============================================================
frames_data = []
S_temp = S_START.copy()
t_emit_temp = 0.0
t_temp = 0.0
path_temp = [S_temp.copy()]
green_temp = []
M_temp = []
last_theta_temp = None

# 跳过静默期
t_temp = 3.0
t_emit_temp = 0.0

while t_temp < MAX_TIME:
    t_temp += DT
    R_S = V_WAVE * t_temp
    R_E = V_WAVE * (t_temp - t_emit_temp)
    d = np.linalg.norm(E - S_temp)

    if d < 0.3:
        break

    if R_S - R_E >= d and d > 0:
        t_emit_temp = t_temp
        R_E = V_WAVE * (t_temp - t_emit_temp)

    O, R = compute_green_circle(S_temp, E, R_S, R_E)

    if O is not None and R > 0:
        green_temp.append((O.copy(), R, t_temp))
        M, theta, forbidden = select_M(O, R, E, BETA, OBSTACLES, last_theta_temp)
        last_theta_temp = theta
        M_temp.append(M.copy())
        move_dir = M - S_temp
        move_dist = np.linalg.norm(move_dir)
        if move_dist > 1e-9:
            actual_step = min(STEP_SIZE, move_dist)
            S_temp = S_temp + (move_dir / move_dist) * actual_step
        path_temp.append(S_temp.copy())
        current_forbidden = forbidden
    else:
        current_forbidden = []
        move_dir = E - S_temp
        move_dist = np.linalg.norm(move_dir)
        if move_dist > 1e-9:
            actual_step = min(STEP_SIZE * 0.5, move_dist)
            S_temp = S_temp + (move_dir / move_dist) * actual_step
        path_temp.append(S_temp.copy())

    frames_data.append({
        't': t_temp,
        'S': S_temp.copy(),
        'R_S': V_WAVE * t_temp,
        'R_E': V_WAVE * (t_temp - t_emit_temp),
        'O': O,
        'R': R,
        'green_count': len(green_temp),
        'd': d,
        'M': M_temp[-1] if M_temp else None,
        'M_all': [m.copy() for m in M_temp],
        'forbidden': current_forbidden,
    })
# ============================================================
# 绘图设置
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ---- 上图：绿色圆波动 + 虚轴避障 + M 点 ----
ax1.set_xlim(-8, 8)
ax1.set_ylim(-6, 6)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('上图：虚轴避障选取 M 点（β 限制旋转范围）', fontsize=14, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax1.plot(S_START[0], S_START[1], 'ko', markersize=10, label='起点 A')
ax1.plot(E[0], E[1], 'k*', markersize=18, label='终点 B')
ax1.axhline(y=0, color='gray', alpha=0.2, linewidth=0.5)

for obs in OBSTACLES:
    circle = Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.4, zorder=3)
    ax1.add_patch(circle)
    ax1.plot(obs[0], obs[1], 'x', color='darkred', markersize=8)

S_point_1, = ax1.plot([], [], 'bo', markersize=10, zorder=5)
S_wave_1 = Circle((0, 0), 0, fill=False, color='blue', alpha=0.3, linewidth=1.5, linestyle='--')
E_wave_1 = Circle((0, 0), 0, fill=False, color='red', alpha=0.3, linewidth=1.5, linestyle='--')
ax1.add_patch(S_wave_1)
ax1.add_patch(E_wave_1)

green_circles_1 = []
path_line_1, = ax1.plot([], [], 'b-', alpha=0.5, linewidth=2)

legend1 = [
    mpatches.Patch(color='blue', alpha=0.3, label='S 波前'),
    mpatches.Patch(color='red', alpha=0.3, label='E 波前'),
    mpatches.Patch(color='green', alpha=0.4, label='绿色圆'),
    mpatches.Patch(color='magenta', alpha=0.5, label='虚轴 + M 点'),
    mpatches.Patch(color='blue', alpha=0.5, label='禁行弧区'),
    mpatches.Patch(color='red', alpha=0.4, label='障碍物'),
]
ax1.legend(handles=legend1, loc='upper right', fontsize=8)

info_text_1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ---- 下图：M 点连线绕过障碍物 ----
ax2.set_xlim(-8, 8)
ax2.set_ylim(-6, 6)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('下图：M₁ → M₂ → M₃ → ... → 绕过障碍物 → 终点收敛', fontsize=14, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

ax2.plot(S_START[0], S_START[1], 'ko', markersize=10, label='起点 A')
ax2.plot(E[0], E[1], 'k*', markersize=18, label='终点 B')
ax2.axhline(y=0, color='gray', alpha=0.2, linewidth=0.5)

for obs in OBSTACLES:
    circle = Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.4, zorder=3)
    ax2.add_patch(circle)
    ax2.plot(obs[0], obs[1], 'x', color='darkred', markersize=8)

S_point_2, = ax2.plot([], [], 'bo', markersize=10, zorder=5)
path_line_2, = ax2.plot([], [], 'orange', alpha=0.8, linewidth=2.5, label='S 轨迹')

green_circles_2 = []

legend2 = [
    mpatches.Patch(color='orange', alpha=0.6, label='S 轨迹'),
    mpatches.Patch(color='magenta', alpha=0.5, label='M 点连线 (路径)'),
    mpatches.Patch(color='green', alpha=0.3, label='绿色圆（收敛）'),
    mpatches.Patch(color='red', alpha=0.4, label='障碍物'),
]
ax2.legend(handles=legend2, loc='upper right', fontsize=8)

info_text_2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
# ============================================================
# 动画函数
# ============================================================
def init():
    S_point_1.set_data([], [])
    S_point_2.set_data([], [])
    path_line_1.set_data([], [])
    path_line_2.set_data([], [])
    info_text_1.set_text('')
    info_text_2.set_text('')
    return (S_point_1, S_point_2, path_line_1, path_line_2,
            info_text_1, info_text_2, S_wave_1, E_wave_1)
def animate(frame):
    global green_circles_1, green_circles_2
    if frame >= len(frames_data):
        return (S_point_1, S_point_2, path_line_1, path_line_2,
                info_text_1, info_text_2, S_wave_1, E_wave_1)
    data = frames_data[frame]
    S_curr = data['S']
    R_S = data['R_S']
    R_E = data['R_E']
    O = data['O']
    R = data['R']
    t_curr = data['t']
    d = data['d']
    green_count = data['green_count']
    M_curr = data['M']
    M_all = data['M_all']
    forbidden = data['forbidden']
    # ---- 上图 ----
    S_point_1.set_data([S_curr[0]], [S_curr[1]])
    S_wave_1.set_center((S_curr[0], S_curr[1]))
    S_wave_1.set_radius(R_S)
    E_wave_1.set_center((E[0], E[1]))
    E_wave_1.set_radius(max(R_E, 0))
    for patch in green_circles_1:
        patch.remove()
    green_circles_1.clear()
    if O is not None and R is not None and R > 0:
        # 绿色圆
        patch = Circle((O[0], O[1]), R, fill=False, color='limegreen',
                       alpha=0.9, linewidth=3)
        ax1.add_patch(patch)
        green_circles_1.append(patch)
        # 禁行弧区（蓝色）
        for low, high in forbidden:
            arc = Arc((O[0], O[1]), 2*R, 2*R, angle=0,
                      theta1=np.degrees(low), theta2=np.degrees(high),
                      color='blue', alpha=0.5, linewidth=4)
            ax1.add_patch(arc)
            green_circles_1.append(arc)
        # 虚轴（紫色线从 O 到 M）
        if M_curr is not None:
            dir_vec = M_curr - O
            if np.linalg.norm(dir_vec) > 1e-9:
                dir_unit = dir_vec / np.linalg.norm(dir_vec)
                p1 = O - R * dir_unit
                p2 = O + R * dir_unit
                v_line, = ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'm--', alpha=0.7, linewidth=2)
                green_circles_1.append(v_line)
        # M 点
        if M_curr is not None:
            M_dot, = ax1.plot(M_curr[0], M_curr[1], 'mo', markersize=10, zorder=6)
            green_circles_1.append(M_dot)
        # 历史绿色圆波动（用当前绿色圆的缩放比例模拟）
        for i in range(1, 5):
            scale = 0.4 + 0.15 * i
            alpha_val = 0.08 * i
            hist_R = R * scale
            patch = Circle((O[0], O[1]), hist_R, fill=False, color='green',
                   alpha=alpha_val, linewidth=1)
            ax1.add_patch(patch)
            green_circles_1.append(patch)

    path_arr = np.array([fd['S'] for fd in frames_data[:frame+1]])
    path_line_1.set_data(path_arr[:, 0], path_arr[:, 1])

    info_text_1.set_text(
        f"⏱ t={t_curr:.2f}s | R_S={R_S:.2f} | R_E={R_E:.2f}\n"
        f"📍 S=({S_curr[0]:.2f}, {S_curr[1]:.2f}) | d={d:.2f}\n"
        f"🟣 M 点数: {len(M_all)} | β={np.degrees(BETA):.0f}°"
    )

    # ---- 下图 ----
    S_point_2.set_data([S_curr[0]], [S_curr[1]])
    path_line_2.set_data(path_arr[:, 0], path_arr[:, 1])

    for patch in green_circles_2:
        patch.remove()
    green_circles_2.clear()

    # 绿色圆收敛
    total_frames = len(frames_data)
    num_display = 10
    indices = np.linspace(0, frame, min(num_display, frame + 1), dtype=int)

    for i, idx in enumerate(indices):
        d_data = frames_data[idx]
        d_O = d_data['O']
        d_R = d_data['R']
        if d_O is not None and d_R is not None and d_R > 0:
            alpha_val = 0.08 + 0.4 * (i / max(1, len(indices) - 1)) if len(indices) > 1 else 0.5
            color_val = plt.cm.Greens(0.3 + 0.7 * (i / max(1, len(indices) - 1))) if len(indices) > 1 else 'limegreen'
            patch = Circle((d_O[0], d_O[1]), d_R, fill=False,
                           color=color_val, alpha=alpha_val, linewidth=1.5)
            ax2.add_patch(patch)
            green_circles_2.append(patch)

    if O is not None and R is not None and R > 0:
        patch = Circle((O[0], O[1]), R, fill=False, color='limegreen',
                       alpha=0.9, linewidth=3)
        ax2.add_patch(patch)
        green_circles_2.append(patch)

    # M 点连线（紫色虚线 + 标注）
    if len(M_all) >= 2:
        M_arr = np.array(M_all)
        M_line, = ax2.plot(M_arr[:, 0], M_arr[:, 1], 'm--', alpha=0.6, linewidth=2)
        green_circles_2.append(M_line)
        M_dots, = ax2.plot(M_arr[:, 0], M_arr[:, 1], 'mo', markersize=4, alpha=0.7)
        green_circles_2.append(M_dots)
        # 标注部分 M 点
        step = max(1, len(M_arr) // 6)
        for i in range(0, len(M_arr), step):
            mx, my = M_arr[i]
            label = ax2.text(mx + 0.2, my + 0.2, f'M{i+1}', fontsize=7,
                            color='magenta', alpha=0.8, fontweight='bold')
            green_circles_2.append(label)
        # 最后一个 M 点
        if (len(M_arr) - 1) % step != 0:
            mx, my = M_arr[-1]
            label = ax2.text(mx + 0.2, my + 0.2, f'M{len(M_arr)}', fontsize=7,
                            color='magenta', alpha=0.8, fontweight='bold')
            green_circles_2.append(label)

    info_text_2.set_text(
        f"⏱ t={t_curr:.2f}s | M 点数: {len(M_all)}\n"
        f"📏 距离终点: {d:.2f}\n"
        f"🟣 M₁ → M₂ → ... → M_n → E | β={np.degrees(BETA):.0f}°"
    )

    return (S_point_1, S_point_2, path_line_1, path_line_2,
            info_text_1, info_text_2, S_wave_1, E_wave_1)

# ============================================================
# 运行
# ============================================================
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(frames_data),
                     interval=DT * 1000, blit=False, repeat=True)

plt.tight_layout()
plt.show()
