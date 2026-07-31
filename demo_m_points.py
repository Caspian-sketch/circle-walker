"""
Circle Walker - M 点路径演示
绿色圆上的 M 点序列 M1 → M2 → M3 → ... 形成通往终点的路线
上图：绿色圆波动 + M 点动态选取
下图：M 点连线收敛到终点
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
# ============================================================
# 参数设置
# ============================================================
S_START = np.array([-5.0, 0.0])
E = np.array([5.0, 0.0])
V_WAVE = 2.5
DT = 0.05
STEP_SIZE = 0.08
MAX_TIME = 15.0
OBSTACLES = []
# ============================================================
# 全局状态
# ============================================================
S = S_START.copy()
t_emit = 0.0
t = 0.0
path = [S.copy()]
green_circles_history = []
M_points = []
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
        target_dir = E - O
        theta_target = np.arctan2(target_dir[1], target_dir[0])
        M = O + R * np.array([np.cos(theta_target), np.sin(theta_target)])
        M_temp.append(M.copy())
        move_dir = E - S_temp
        move_dist = np.linalg.norm(move_dir)
        if move_dist > 1e-9:
            actual_step = min(STEP_SIZE, move_dist)
            S_temp = S_temp + (move_dir / move_dist) * actual_step
        path_temp.append(S_temp.copy())
    else:
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
    })
# ============================================================
# 绘图设置
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# ---- 上图：绿色圆波动 + M 点 ----
ax1.set_xlim(-8, 8)
ax1.set_ylim(-6, 6)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('上图：绿色圆波动 + M 点动态选取', fontsize=14, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.plot(S_START[0], S_START[1], 'ko', markersize=10, label='起点 A')
ax1.plot(E[0], E[1], 'k*', markersize=18, label='终点 B')
ax1.axhline(y=0, color='gray', alpha=0.2, linewidth=0.5)
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
    mpatches.Patch(color='magenta', alpha=0.5, label='M 点路径'),
]
ax1.legend(handles=legend1, loc='upper right', fontsize=9)
info_text_1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
# ---- 下图：M 点连线收敛 ----
ax2.set_xlim(-8, 8)
ax2.set_ylim(-6, 6)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('下图：M₁ → M₂ → M₃ → ... → 终点收敛', fontsize=14, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.plot(S_START[0], S_START[1], 'ko', markersize=10, label='起点 A')
ax2.plot(E[0], E[1], 'k*', markersize=18, label='终点 B')
ax2.axhline(y=0, color='gray', alpha=0.2, linewidth=0.5)
S_point_2, = ax2.plot([], [], 'bo', markersize=10, zorder=5)
path_line_2, = ax2.plot([], [], 'orange', alpha=0.8, linewidth=2.5, label='S 轨迹')
green_circles_2 = []
legend2 = [
    mpatches.Patch(color='orange', alpha=0.6, label='S 轨迹'),
    mpatches.Patch(color='magenta', alpha=0.5, label='M 点连线 (路径)'),
    mpatches.Patch(color='green', alpha=0.3, label='绿色圆（收敛）'),
]
ax2.legend(handles=legend2, loc='upper right', fontsize=9)
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
        start_idx = max(0, green_count - 8)
        for i in range(start_idx, green_count):
            scale = 0.3 + 0.7 * ((i - start_idx) / max(1, green_count - start_idx - 1)) if green_count > start_idx + 1 else 1.0
            alpha_val = 0.15 + 0.25 * ((i - start_idx) / max(1, green_count - start_idx))
            hist_R = R * scale
            patch = Circle((O[0], O[1]), hist_R, fill=False, color='green',
                           alpha=alpha_val, linewidth=1.5)
            ax1.add_patch(patch)
            green_circles_1.append(patch)
    # M 点和连线
    if M_curr is not None:
        M_dot, = ax1.plot(M_curr[0], M_curr[1], 'mo', markersize=8, zorder=6)
        green_circles_1.append(M_dot)
        M_label = ax1.text(M_curr[0] + 0.3, M_curr[1] + 0.3, 'M', fontsize=9,
                           color='magenta', fontweight='bold')
        green_circles_1.append(M_label)
    if len(M_all) >= 2:
        M_arr = np.array(M_all)
        M_line, = ax1.plot(M_arr[:, 0], M_arr[:, 1], 'm--', alpha=0.5, linewidth=1.5)
        green_circles_1.append(M_line)
        M_dots, = ax1.plot(M_arr[:, 0], M_arr[:, 1], 'mo', markersize=3, alpha=0.6)
        green_circles_1.append(M_dots)
    path_arr = np.array([fd['S'] for fd in frames_data[:frame+1]])
    path_line_1.set_data(path_arr[:, 0], path_arr[:, 1])
    info_text_1.set_text(
        f"⏱ t={t_curr:.2f}s | R_S={R_S:.2f} | R_E={R_E:.2f}\n"
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
    num_display = 12
    indices = np.linspace(0, frame, min(num_display, frame + 1), dtype=int)
    for i, idx in enumerate(indices):
        d_data = frames_data[idx]
        d_O = d_data['O']
        d_R = d_data['R']
        if d_O is not None and d_R is not None and d_R > 0:
            alpha_val = 0.1 + 0.5 * (i / max(1, len(indices) - 1)) if len(indices) > 1 else 0.6
            color_val = plt.cm.Greens(0.3 + 0.7 * (i / max(1, len(indices) - 1))) if len(indices) > 1 else 'limegreen'
            patch = Circle((d_O[0], d_O[1]), d_R, fill=False,
                           color=color_val, alpha=alpha_val, linewidth=2)
            ax2.add_patch(patch)
            green_circles_2.append(patch)
    if O is not None and R is not None and R > 0:
        patch = Circle((O[0], O[1]), R, fill=False, color='limegreen',
                       alpha=0.9, linewidth=3)
        ax2.add_patch(patch)
        green_circles_2.append(patch)
    # M 点连线
    if len(M_all) >= 2:
        M_arr = np.array(M_all)
        M_line, = ax2.plot(M_arr[:, 0], M_arr[:, 1], 'm--', alpha=0.6, linewidth=2)
        green_circles_2.append(M_line)
        M_dots, = ax2.plot(M_arr[:, 0], M_arr[:, 1], 'mo', markersize=4, alpha=0.7)
        green_circles_2.append(M_dots)
        # M1, M2, M3 标注
        for i, (mx, my) in enumerate(M_arr):
            if i % max(1, len(M_arr)//5) == 0 or i == len(M_arr)-1:
                label = ax2.text(mx + 0.2, my + 0.2, f'M{i+1}', fontsize=7,
                                color='magenta', alpha=0.8)
                green_circles_2.append(label)
    info_text_2.set_text(
        f"⏱ t={t_curr:.2f}s | M 点数: {len(M_all)}\n"
        f"📏 距离终点: {d:.2f} | S → M₁ → M₂ → ... → E"
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
