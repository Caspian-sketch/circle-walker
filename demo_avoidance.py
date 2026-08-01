"""
Circle Walker - 拐弯路径动态生成演示
展示障碍物投影、虚轴偏转、步态跳跃如何协同工作绕过障碍物
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge, Arc
import matplotlib.patches as mpatches
# ============================================================
# 场景设置：起点(A)左，终点(B)右，中间大障碍物
# ============================================================
S_START = np.array([-5.0, 0.0])     # 起点 A
E = np.array([5.0, 0.0])            # 终点 B
OBSTACLES = [
    (1.0, 0.0, 0.2),   # 正中间大障碍物，彻底堵死直线
]
# ============================================================
# 参数设置
# ============================================================
V_WAVE = 4.0                        # 波速（稍快，让绿圆快速形成）
V_AGENT = 0.04                       # S 移动速度
STEP_SIZE = 0.25                    # 步长（较小，展示精细的步态跳跃）
BETA = np.radians(55)              # 虚轴最大偏角（稍大，允许绕行）
DT = 0.04                          # 时间步长
EPSILON = 0.2                      # 收敛阈值
# ============================================================
# 全局状态
# ============================================================
S = S_START.copy()
t_emit = 0.0
t = 0.0
path = [S.copy()]
rebuild_times = []
green_circles_history = []
# 用于动画展示的临时变量
current_O = None
current_R = None
current_M = None
current_forbidden_arcs = []
current_theta_virtual = None
# ============================================================
# 辅助函数
# ============================================================
def compute_green_circle(S, E, R_S, R_E):
    """计算绿色圆的圆心和半径"""
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
    """将障碍物投影到绿色圆上，返回禁入角度区间 (low, high)"""
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
        theta_obs = np.arctan2(d_vec[1], d_vec[0])
        low = theta_obs - alpha
        high = theta_obs + alpha
        if low < -np.pi:
            low += 2 * np.pi
        if high > np.pi:
            high -= 2 * np.pi
        return (low, high)
    alpha = np.arcsin(np.clip(r / d, -1, 1))
    theta_obs = np.arctan2(d_vec[1], d_vec[0])
    low = theta_obs - alpha
    high = theta_obs + alpha
    # 规范化 low 和 high 到 [-pi, pi]
    low = (low + np.pi) % (2 * np.pi) - np.pi
    high = (high + np.pi) % (2 * np.pi) - np.pi
    return (low, high)
def select_M(O, R, E, beta, obstacles):
    """虚轴法选择 M 点，返回 M 和选中的角度 theta"""
    target_dir = E - O
    theta_target = np.arctan2(target_dir[1], target_dir[0])
    
    # 收集所有禁入区间
    forbidden = []
    for obs in obstacles:
        result = project_obstacle(O, R, obs)
        if result is not None:
            forbidden.append(result)

    # 在允许区间内搜索
    search_angles = np.linspace(theta_target - beta, theta_target + beta, 300)
    
    best_theta = theta_target
    best_dist = float('inf')
    
    for theta in search_angles:
        in_forbidden = False
        for low, high in forbidden:
            # 处理角度跨越 ±π 的情况
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
    
    # 如果都被禁了，扩大搜索
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
    
    M = O + R * np.array([np.cos(best_theta), np.sin(best_theta)])
    return M, best_theta, forbidden
# ============================================================
# 绘图设置
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(-7, 7)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Circle Walker — 拐弯路径动态生成', fontsize=16, fontweight='bold')
# 背景装饰
ax.axhline(y=0, color='gray', alpha=0.3, linewidth=0.5)
ax.axvline(x=0, color='gray', alpha=0.3, linewidth=0.5)
# 起点和终点
ax.plot(S_START[0], S_START[1], 'ko', markersize=14, label='起点 A')
ax.plot(E[0], E[1], 'k*', markersize=22, label='终点 B')
# 标签
ax.text(S_START[0]-0.8, S_START[1]+0.4, 'A (起点)', fontsize=12, fontweight='bold')
ax.text(E[0]+0.4, E[1]+0.4, 'B (终点)', fontsize=12, fontweight='bold')
# 障碍物
for obs in OBSTACLES:
    circle = Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5, zorder=3)
    ax.add_patch(circle)
    ax.plot(obs[0], obs[1], 'x', color='darkred', markersize=10, markeredgewidth=2)
    ax.text(obs[0]+0.3, obs[1]-0.3, '障碍物', fontsize=10, color='darkred')
# 连线提示
ax.annotate('', xy=(E[0]-0.5, E[1]), xytext=(S_START[0]+0.5, S_START[1]),
            arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.4, linewidth=1))
ax.text(0, 0.8, '直线路径被堵死', fontsize=10, color='gray', ha='center')
# 动态元素
S_point, = ax.plot([], [], 'bo', markersize=12, zorder=5, label='智能体 S')
S_wave = Circle((0, 0), 0, fill=False, color='blue', alpha=0.3, linewidth=1.5, linestyle='--')
E_wave = Circle((0, 0), 0, fill=False, color='red', alpha=0.3, linewidth=1.5, linestyle='--')
green_circle_patch = Circle((0, 0), 0, fill=False, color='green', alpha=0.7, linewidth=2.5)
ax.add_patch(S_wave)
ax.add_patch(E_wave)
ax.add_patch(green_circle_patch)
# 障碍物投影弧（蓝色弧线）
forbidden_arcs = []
# 虚轴（紫色虚线）
virtual_line, = ax.plot([], [], 'm--', alpha=0.8, linewidth=2, label='虚轴')
# M 点
M_point, = ax.plot([], [], 'mo', markersize=10, zorder=5, label='M 点 (下一步)')
# 轨迹（橙色步态跳跃）
path_line, = ax.plot([], [], color='orange', alpha=0.9, linewidth=2.5, 
                     marker='.', markersize=3, label='步态轨迹')
# 历史绿色圆（半透明）
history_patches = []
# 图例
legend_elements = [
    mpatches.Patch(color='blue', alpha=0.3, label='S 波前'),
    mpatches.Patch(color='red', alpha=0.3, label='E 波前'),
    mpatches.Patch(color='green', alpha=0.4, label='绿色圆'),
    mpatches.Patch(color='magenta', alpha=0.6, label='虚轴'),
    mpatches.Patch(color='orange', alpha=0.6, label='步态轨迹'),
    mpatches.Patch(color='red', alpha=0.5, label='障碍物'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
# 信息面板
info_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                   fontfamily='monospace', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
# ============================================================
# 动画函数
# ============================================================
def init():
    S_point.set_data([], [])
    virtual_line.set_data([], [])
    M_point.set_data([], [])
    path_line.set_data([], [])
    info_box.set_text('')
    return (S_point, virtual_line, M_point, path_line, info_box,
            S_wave, E_wave, green_circle_patch)
def animate(frame):
    global S, t_emit, t, path, rebuild_times, green_circles_history
    global current_O, current_R, current_M, current_forbidden_arcs, current_theta_virtual
    if t == 0.0:
        t = 3.0
        t_emit = 0.0
    t += DT
    
    R_S = V_WAVE * t
    R_E = V_WAVE * (t - t_emit)
    
    d = np.linalg.norm(E - S)
    
    # 收敛检查
    if d < EPSILON:
        S = E.copy()
        path.append(S.copy())
        update_visuals(frame)
        return (S_point, virtual_line, M_point, path_line, info_box,
                S_wave, E_wave, green_circle_patch)
    
    # 事件驱动重建
    if R_S - R_E >= d and d > 0:
        t_emit = t
        R_E = 0
        rebuild_times.append(t)
    
    R_E = V_WAVE * (t - t_emit)
    
    # 计算绿色圆
    O, R = compute_green_circle(S, E, R_S, R_E)

    if O is not None and R > 0:
        # === 闭环阶段：绿色圆存在，虚轴避障 ===
        current_O = O
        current_R = R
        green_circles_history.append((O.copy(), R, t))
    
        # 虚轴选点
        M, theta, forbidden = select_M(O, R, E, BETA, OBSTACLES)
        if forbidden:
            for low, high in forbidden:
                print(f"t={t:.2f} 禁入区间: [{np.degrees(low):.1f}°, {np.degrees(high):.1f}°] 目标方向: {np.degrees(np.arctan2(E[1]-O[1], E[0]-O[0])):.1f}°")
        current_M = M
        current_theta_virtual = theta
        current_forbidden_arcs = forbidden
    
        # S 向 M 移动
        move_dir = M - S
        move_dist = np.linalg.norm(move_dir)
        if move_dist > 1e-9:
            actual_step = min(0.1, move_dist)
            S = S + (move_dir / move_dist) * actual_step
        path.append(S.copy())
    else:
        # === 静默期：绿色圆尚未形成，直接朝终点缓慢移动 ===
        current_O = None
        current_R = None
        current_M = None
        current_forbidden_arcs = []
        move_dir = E - S
        move_dist = np.linalg.norm(move_dir)
        if move_dist > 1e-9:
            actual_step = min(0.01, move_dist)
            S = S + (move_dir / move_dist) * actual_step
        path.append(S.copy())
    update_visuals(frame)
    
    return (S_point, virtual_line, M_point, path_line, info_box,
            S_wave, E_wave, green_circle_patch)
def update_visuals(frame):
    """更新所有可视化元素"""
    R_S = V_WAVE * t
    R_E = V_WAVE * (t - t_emit)
    d = np.linalg.norm(E - S)
    
    # S 点
    S_point.set_data([S[0]], [S[1]])
    
    # 波前圆
    S_wave.set_center((S[0], S[1]))
    S_wave.set_radius(R_S)
    E_wave.set_center((E[0], E[1]))
    E_wave.set_radius(max(R_E, 0))
    
    # 绿色圆
    if current_O is not None and current_R is not None and current_R > 0:
        green_circle_patch.set_center((current_O[0], current_O[1]))
        green_circle_patch.set_radius(current_R)
        green_circle_patch.set_visible(True)
        
        # 障碍物投影弧（蓝色）
        global forbidden_arcs
        for arc in forbidden_arcs:
            arc.remove()
        forbidden_arcs.clear()
        
        for low, high in current_forbidden_arcs:
            # 在绿色圆上画出禁行弧区
            arc = Arc((current_O[0], current_O[1]), 
                     2*current_R, 2*current_R,
                     angle=0, theta1=np.degrees(low), theta2=np.degrees(high),
                     color='blue', alpha=0.5, linewidth=4)
            ax.add_patch(arc)
            forbidden_arcs.append(arc)
        
        # 虚轴
        if current_M is not None:
            dir_vec = current_M - current_O
            if np.linalg.norm(dir_vec) > 1e-9:
                dir_unit = dir_vec / np.linalg.norm(dir_vec)
                p1 = current_O - current_R * dir_unit
                p2 = current_O + current_R * dir_unit
                virtual_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        # M 点
        if current_M is not None:
            M_point.set_data([current_M[0]], [current_M[1]])
    else:
        green_circle_patch.set_visible(False)
        virtual_line.set_data([], [])
        M_point.set_data([], [])
    
    # 轨迹
    path_arr = np.array(path)
    path_line.set_data(path_arr[:, 0], path_arr[:, 1])
    
    # 历史绿色圆
    global history_patches
    for patch in history_patches:
        patch.remove()
    history_patches.clear()
    
    if len(green_circles_history) > 0:
        step = max(1, len(green_circles_history) // 20)
        for i in range(0, len(green_circles_history), step):
            O_hist, R_hist, t_hist = green_circles_history[i]
            alpha_val = 0.05 + 0.1 * (i / max(1, len(green_circles_history)))
            patch = Circle((O_hist[0], O_hist[1]), R_hist,
                          fill=False, color='green', alpha=alpha_val, linewidth=1)
            ax.add_patch(patch)
            history_patches.append(patch)
    
    # 信息面板
    info_str = (
        f"⏱ 时间: {t:.2f}s | 重建次数: {len(rebuild_times)}\n"
        f"📡 R_S: {R_S:.2f} | R_E: {R_E:.2f}\n"
        f"📏 S-E 距离: {d:.2f}\n"
        f"📍 S: ({S[0]:.2f}, {S[1]:.2f})"
    )
    if current_O is not None:
        info_str += f"\n🟢 绿圆 O: ({current_O[0]:.2f}, {current_O[1]:.2f}) R: {current_R:.2f}"
        info_str += f"\n🟣 虚轴角度: {np.degrees(current_theta_virtual):.1f}°"
        if current_forbidden_arcs:
            info_str += f"\n🔵 禁行弧区: {len(current_forbidden_arcs)} 段"
    info_box.set_text(info_str)
# ============================================================
# 运行
# ============================================================
anim = FuncAnimation(fig, animate, init_func=init, frames=None,
                     interval=DT * 1000, blit=False, repeat=False)
plt.tight_layout()
plt.show()
