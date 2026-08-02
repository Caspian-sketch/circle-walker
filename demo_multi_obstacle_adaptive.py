"""
Circle Walker - 自适应半径演示
通过半径缩放因子 k_r 动态调节 M 点到 S 的距离，
解决起步脱节、引导滞后、障碍物处路径分离等问题。
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
V_WAVE = 2.0
DT = 0.01
STEP_SIZE = 0.1
BETA = np.radians(55)
MAX_TIME = 25.0

# 半径缩放因子
K_R_MIN = 0.3
K_R_MAX = 1.0
K_R = 0.5          # 初始缩放因子

# 障碍物
OBSTACLES = [
    (1.0, 0.0, 0.5),
    (1.5,-0.6,0.5),
]

# ============================================================
# 全局状态
# ============================================================
S = S_START.copy()
t_emit = 0.0
t = 0.0
path = [S.copy()]
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


def find_available_intervals(theta_target, beta, forbidden):
    intervals = []
    search_start = theta_target - beta
    search_end = theta_target + beta
    sorted_forbidden = []
    for low, high in forbidden:
        if low <= high:
            sorted_forbidden.append((low, high))
        else:
            sorted_forbidden.append((low, np.pi))
            sorted_forbidden.append((-np.pi, high))
    sorted_forbidden.sort()
    current = search_start
    for low, high in sorted_forbidden:
        if low > current:
            intervals.append((current, min(low, search_end)))
        current = max(current, high)
        if current >= search_end:
            break
    if current < search_end:
        intervals.append((current, search_end))
    return intervals
def select_M(O, R, E, beta, obstacles, last_theta, k_r):
    target_dir = E - O
    theta_target = np.arctan2(target_dir[1], target_dir[0])

    forbidden = []
    for obs in obstacles:
        result = project_obstacle(O, R, obs)
        if result is not None:
            forbidden.append(result)

    num_samples = 300
    raw_angles = np.linspace(theta_target - beta, theta_target + beta, num_samples)
    search_angles = (raw_angles + np.pi) % (2 * np.pi) - np.pi

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
            diff = abs((theta - theta_target + np.pi) % (2 * np.pi) - np.pi)
            if diff < best_dist:
                best_dist = diff
                best_theta = theta

    if best_dist == float('inf'):
        wider_angles = np.linspace(theta_target - np.pi/2, theta_target + np.pi/2, 500)
        wider_angles = (wider_angles + np.pi) % (2 * np.pi) - np.pi
        for theta in wider_angles:
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
                diff = abs((theta - theta_target + np.pi) % (2 * np.pi) - np.pi)
                if diff < best_dist:
                    best_dist = diff
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
        diff = (best_theta - last_theta + np.pi) % (2 * np.pi) - np.pi
        if old_available and abs(diff) < np.radians(5):
            best_theta = last_theta

    # 可用区间宽度
    intervals = find_available_intervals(theta_target, beta, forbidden)
    if intervals:
        available_width = max(high - low for low, high in intervals)
    else:
        available_width = 0.0

    # 二维扫描：在绿色圆内搜索 M 点
    best_M = None
    best_dist_to_E = float('inf')
    radii = np.linspace(0.3 * R * k_r, R * k_r, 20)

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
        if in_forbidden:
            continue

        for r in reversed(radii):
            candidate = O + r * np.array([np.cos(theta), np.sin(theta)])
            safe = True
            for obs in obstacles:
                obs_center = np.array([obs[0], obs[1]])
                if np.linalg.norm(candidate - obs_center) < obs[2]:
                    safe = False
                    break
            if safe:
                dist_to_E = np.linalg.norm(candidate - E)
                if dist_to_E < best_dist_to_E:
                    best_dist_to_E = dist_to_E
                    best_M = candidate
                break

    if best_M is not None:
        M = best_M
        best_theta = np.arctan2(M[1] - O[1], M[0] - O[0])
    else:
        R_scaled = R * k_r
        M = O + R_scaled * np.array([np.cos(best_theta), np.sin(best_theta)])

    # 限制 M 点不超过终点
    S_to_E = E - O
    dist_O_to_E = np.linalg.norm(S_to_E)
    if np.linalg.norm(M - O) > dist_O_to_E:
        if np.dot(M - O, S_to_E) > dist_O_to_E**2:
            M = E.copy()

    return M, best_theta, forbidden, available_width
# ============================================================
# 预计算所有帧的数据
# ============================================================
frames_data = []
S_temp = S_START.copy()
t_emit_temp = 0.0
t_temp = 0.0
path_temp = [S_temp.copy()]
M_temp = []
last_theta_temp = None
k_r = K_R

# 跳过静默期
t_temp = 2.5
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
        # 初始阶段：把圆心 O 往 S 拉近，让 M₁ 从 S 身边起步
        if len(M_temp) == 0:
            dist_S_to_O = np.linalg.norm(O - S_temp)
            if dist_S_to_O > 1.0:
                O = S_temp + (O - S_temp) / dist_S_to_O * 1.0
                R = R * (0.5 / dist_S_to_O)
        # 动态调节 k_r
        if len(M_temp) > 0:
            dist_to_last_M = np.linalg.norm(S_temp - M_temp[-1])
            if dist_to_last_M < 0.15:
                k_r = min(k_r + 0.03, K_R_MAX)
            elif dist_to_last_M > 0.5:
                k_r = max(k_r - 0.05, K_R_MIN)

        M, theta, forbidden, available_width = select_M(
            O, R, E, BETA, OBSTACLES, last_theta_temp, k_r
        )
        last_theta_temp = theta
        M_temp.append(M.copy())

        # S 向 M 移动
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
            actual_step = min(0.02, move_dist)
            S_temp = S_temp + (move_dir / move_dist) * actual_step
        path_temp.append(S_temp.copy())

    frames_data.append({
        't': t_temp,
        'S': S_temp.copy(),
        'R_S': V_WAVE * t_temp,
        'R_E': V_WAVE * (t_temp - t_emit_temp),
        'O': O,
        'R': R,
        'd': d,
        'M': M_temp[-1] if len(M_temp) > 0 else None,
        'M_all': [m.copy() for m in M_temp],
        'forbidden': current_forbidden,
        'k_r': k_r,
    })
# ============================================================
# 绘图设置
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ---- 上图 ----
ax1.set_xlim(-8, 8)
ax1.set_ylim(-6, 6)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('上图：自适应半径 M 点引导（k_r 动态调节）', fontsize=14, fontweight='bold')
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
ax1.legend(handles=legend1, loc='upper right', fontsize=7)

info_text_1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ---- 下图 ----
ax2.set_xlim(-8, 8)
ax2.set_ylim(-6, 6)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('下图：橙色路径与紫色 M 点同步收敛', fontsize=14, fontweight='bold')
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
    mpatches.Patch(color='magenta', alpha=0.5, label='M 点连线'),
    mpatches.Patch(color='green', alpha=0.3, label='绿色圆'),
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
    M_curr = data['M']
    M_all = data['M_all']
    forbidden = data['forbidden']
    k_r = data['k_r']
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
        patch = Circle((O[0], O[1]), R, fill=False, color='limegreen',
                       alpha=0.9, linewidth=3)
        ax1.add_patch(patch)
        green_circles_1.append(patch)
        for low, high in forbidden:
            arc = Arc((O[0], O[1]), 2*R, 2*R, angle=0,
                      theta1=np.degrees(low), theta2=np.degrees(high),
                      color='blue', alpha=0.5, linewidth=4)
            ax1.add_patch(arc)
            green_circles_1.append(arc)
        if M_curr is not None:
            dir_vec = M_curr - O
            if np.linalg.norm(dir_vec) > 1e-9:
                dir_unit = dir_vec / np.linalg.norm(dir_vec)
                p1 = O - R * dir_unit
                p2 = O + R * dir_unit
                v_line, = ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'm--', alpha=0.7, linewidth=2)
                green_circles_1.append(v_line)
        if M_curr is not None:
            M_dot, = ax1.plot(M_curr[0], M_curr[1], 'mo', markersize=10, zorder=6)
            green_circles_1.append(M_dot)
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
        f"⏱ t={t_curr:.2f}s | k_r={k_r:.2f}\n"
        f"📍 S=({S_curr[0]:.2f}, {S_curr[1]:.2f}) | d={d:.2f}\n"
        f"🟣 M 点数: {len(M_all)}"
    )
    # ---- 下图 ----
    S_point_2.set_data([S_curr[0]], [S_curr[1]])
    path_line_2.set_data(path_arr[:, 0], path_arr[:, 1])
    for patch in green_circles_2:
        patch.remove()
    green_circles_2.clear()
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
    if len(M_all) >= 2:
        M_arr = np.array(M_all)
        M_line, = ax2.plot(M_arr[:, 0], M_arr[:, 1], 'm--', alpha=0.6, linewidth=2)
        green_circles_2.append(M_line)
        M_dots, = ax2.plot(M_arr[:, 0], M_arr[:, 1], 'mo', markersize=3, alpha=0.6)
        green_circles_2.append(M_dots)
    info_text_2.set_text(
        f"⏱ t={t_curr:.2f}s | k_r={k_r:.2f}\n"
        f"📏 距离终点: {d:.2f} | M 点数: {len(M_all)}"
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
