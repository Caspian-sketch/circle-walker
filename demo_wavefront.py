import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Arc

# ===========================
# 1. 环境与参数设置
# ===========================
A = np.array([-4.0, 0.0])
B = np.array([4.0, 0.0])
v_wave = 1.5
step_size = 0.6  # 步子迈大一点，圆心移动更明显
frames = 100
total_time = 10.0

obstacle = np.array([1.0, 1.5]) 
obstacle_radius = 0.6

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10))
plt.subplots_adjust(hspace=0.3)

def init():
    ax_top.set_title("Top: Wave Interference", fontsize=12)
    ax_top.set_xlim(-7, 7); ax_top.set_ylim(-4, 4); ax_top.set_aspect('equal')
    ax_bot.set_title("Bottom: Green Circle Center 'O' Moves Right as S Jumps", fontsize=12)
    ax_bot.set_xlim(-7, 7); ax_bot.set_ylim(-4, 4); ax_bot.set_aspect('equal')
    return []

def update(frame):
    t = (frame / frames) * total_time
    r_wave = v_wave * t
    
    # 计算步数和当前S的位置
    steps_taken = int((v_wave * t) / step_size) 
    S_curr = np.array([min(A[0] + steps_taken * step_size, B[0] - 0.2), 0])
    
    # 计算当前的绿圆圆心 O_curr
    d_SB = np.linalg.norm(B - S_curr)
    O_curr = (S_curr + B) / 2.0
    
    # ---- 绘制上半图 ----
    ax_top.clear()
    ax_top.set_xlim(-7, 7); ax_top.set_ylim(-4, 4); ax_top.set_aspect('equal')
    ax_top.grid(True, linestyle=':', alpha=0.5)
    ax_top.scatter(A[0], A[1], c='black', s=50, label='Start')
    ax_top.scatter(B[0], B[1], c='black', s=50, label='End')
    ax_top.scatter(obstacle[0], obstacle[1], c='red', s=200, alpha=0.4, label='Obstacle')
    
    circle_A = plt.Circle(A, r_wave, color='blue', fill=False, linestyle='--', lw=1.5, alpha=0.5)
    circle_B = plt.Circle(B, r_wave, color='red', fill=False, linestyle='--', lw=1.5, alpha=0.5)
    ax_top.add_patch(circle_A); ax_top.add_patch(circle_B)
    
    d_AB = np.linalg.norm(B - A)
    if r_wave > d_AB/2:
        O_fixed = (A + B) / 2.0
        r_g = np.sqrt(r_wave**2 - (d_AB/2)**2)
        ax_top.add_patch(plt.Circle(O_fixed, r_g, color='green', fill=False, lw=3))
    ax_top.legend(loc='upper right')

    # ---- 绘制下半图 ----
    ax_bot.clear()
    ax_bot.set_xlim(-7, 7); ax_bot.set_ylim(-4, 4); ax_bot.set_aspect('equal')
    ax_bot.grid(True, linestyle=':', alpha=0.5)
    ax_bot.scatter(A[0], A[1], c='black', s=50); ax_bot.text(A[0]-0.4, A[1]-0.6, 'Start', fontsize=10)
    ax_bot.scatter(B[0], B[1], c='black', s=50); ax_bot.text(B[0]-0.4, B[1]-0.6, 'Goal', fontsize=10)
    ax_bot.scatter(obstacle[0], obstacle[1], c='firebrick', s=400, alpha=0.4)

    # 1. 画历史轨迹上的“绿圆圆心”
    for i in range(steps_taken + 1):
        S_hist = np.array([A[0] + i * step_size, 0])
        O_hist = (S_hist + B) / 2.0
        # 用绿色小十字标出历史上的圆心
        ax_bot.plot(O_hist[0], O_hist[1], marker='+', color='forestgreen', markersize=10, markeredgewidth=2)

    # 2. 画当前智能体S
    ax_bot.scatter(S_curr[0], S_curr[1], c='orange', s=150, edgecolors='black', zorder=10)
    ax_bot.text(S_curr[0]+0.3, S_curr[1], 'S', color='black', fontsize=14, fontweight='bold')

    # 3. 画当前的绿圆
    if t > 0.5 and d_SB < r_wave * 2: 
        r_g_curr = np.sqrt(r_wave**2 - (d_SB/2)**2)
        ax_bot.add_patch(plt.Circle(O_curr, r_g_curr, color='green', fill=False, lw=2, linestyle='--'))
        
        # 标出当前圆心文字
        ax_bot.text(O_curr[0]-0.2, O_curr[1]+0.3, 'O', color='darkgreen', fontsize=12, fontweight='bold')

        # 避障逻辑与 M点计算
        v_obs = obstacle - O_curr
        dist_obs = np.linalg.norm(v_obs)
        theta = t * 1.2 # 基准旋转
        if dist_obs > 0:
            alpha = np.arcsin(obstacle_radius / dist_obs)
            beta = np.arctan2(v_obs[1], v_obs[0])
            start_angle = np.degrees(beta - alpha)
            end_angle = np.degrees(beta + alpha)
            ax_bot.add_patch(Arc(O_curr, r_g_curr*2, r_g_curr*2, theta1=start_angle, theta2=end_angle, 
                      color='blue', lw=4, alpha=0.7))
            # 模拟避开
            if np.sin(theta - beta) > 0:
                theta = beta + alpha + 0.1
            else:
                theta = beta - alpha - 0.1

        M_x = O_curr[0] + r_g_curr * np.cos(theta)
        M_y = O_curr[1] + r_g_curr * np.sin(theta)
        
        ax_bot.plot([O_curr[0], M_x], [O_curr[1], M_y], color='purple', linestyle='-', lw=2, alpha=0.7)
        ax_bot.scatter(M_x, M_y, c='purple', s=120, edgecolors='white', zorder=8)
        ax_bot.text(M_x+0.2, M_y, 'M', color='purple', fontsize=12, fontweight='bold')
        
        # 轨迹
        x_trail = [A[0] + i * step_size for i in range(steps_taken + 1)]
    # 1. 画历史轨迹上的“绿圆圆心”
    for i in range(steps_taken + 1):
        S_hist = np.array([A[0] + i * step_size, 0])
        O_hist = (S_hist + B) / 2.0
        # 用绿色小十字标出历史上的圆心
        ax_bot.plot(O_hist[0], O_hist[1], marker='+', color='forestgreen', markersize=10, markeredgewidth=2)
    # 2. 画当前智能体S
    ax_bot.scatter(S_curr[0], S_curr[1], c='orange', s=150, edgecolors='black', zorder=10)
    ax_bot.text(S_curr[0]+0.3, S_curr[1], 'S', color='black', fontsize=14, fontweight='bold')
    # 3. 画当前的绿圆
    if t > 0.5 and d_SB < r_wave * 2: 
        r_g_curr = np.sqrt(r_wave**2 - (d_SB/2)**2)
        ax_bot.add_patch(plt.Circle(O_curr, r_g_curr, color='green', fill=False, lw=2, linestyle='--'))
        
        # 标出当前圆心文字
        ax_bot.text(O_curr[0]-0.2, O_curr[1]+0.3, 'O', color='darkgreen', fontsize=12, fontweight='bold')
        # 避障逻辑与 M点计算
        v_obs = obstacle - O_curr
        dist_obs = np.linalg.norm(v_obs)
        theta = t * 1.2 # 基准旋转
        if dist_obs > 0:
            alpha = np.arcsin(obstacle_radius / dist_obs)
            beta = np.arctan2(v_obs[1], v_obs[0])
            start_angle = np.degrees(beta - alpha)
            end_angle = np.degrees(beta + alpha)
            ax_bot.add_patch(Arc(O_curr, r_g_curr*2, r_g_curr*2, theta1=start_angle, theta2=end_angle, 
                      color='blue', lw=4, alpha=0.7))
            # 模拟避开
            if np.sin(theta - beta) > 0:
                theta = beta + alpha + 0.1
            else:
                theta = beta - alpha - 0.1
        M_x = O_curr[0] + r_g_curr * np.cos(theta)
        M_y = O_curr[1] + r_g_curr * np.sin(theta)
        
        ax_bot.plot([O_curr[0], M_x], [O_curr[1], M_y], color='purple', linestyle='-', lw=2, alpha=0.7)
        ax_bot.scatter(M_x, M_y, c='purple', s=120, edgecolors='white', zorder=8)
        ax_bot.text(M_x+0.2, M_y, 'M', color='purple', fontsize=12, fontweight='bold')
        
        # 轨迹
        x_trail = [A[0] + i * step_size for i in range(steps_taken + 1)]
        y_trail = [0] * len(x_trail)
        ax_bot.plot(x_trail, y_trail, color='gray', linestyle=':', lw=1, alpha=0.5)
        ax_bot.legend(loc='upper right')
    return []
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=80)
plt.show()
