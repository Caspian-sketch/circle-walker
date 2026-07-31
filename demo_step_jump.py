import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Arc

# ===========================
# 1. 环境与参数设置
# ===========================
A = np.array([-4.0, 0.0])       # 起点
B = np.array([4.0, 0.0])        # 终点
v_wave = 1.8                    # 波前扩散速度
step_size = 0.8                 # 智能体每步移动距离
frames = 120                    
total_time = 10.0               

obstacle = np.array([1.0, 1.5]) 
obstacle_radius = 0.6

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

def init():
    ax.set_title("Circle Walker: S jumps to M (Landing Effect)", fontsize=16)
    ax.set_xlim(-6, 6); ax.set_ylim(-3.5, 3.5); ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)
    return []

def update(frame):
    ax.clear()
    ax.set_title("Circle Walker: S jumps to M (Landing Effect)", fontsize=16)
    ax.set_xlim(-6, 6); ax.set_ylim(-3.5, 3.5); ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # 画固定点
    ax.scatter(A[0], A[1], c='black', s=80, label='Start (A)')
    ax.scatter(B[0], B[1], c='black', s=80, label='Goal (E)')
    ax.scatter(obstacle[0], obstacle[1], c='firebrick', s=500, alpha=0.3, label='Obstacle')

    t = (frame / frames) * total_time
    r_wave = v_wave * t
    
    # 计算当前步数
    steps_taken = int((v_wave * t) / step_size) 
    S_curr = np.array([min(A[0] + steps_taken * step_size, B[0] - 0.2), 0])
    
    d_SB = np.linalg.norm(B - S_curr)
    O_curr = (S_curr + B) / 2.0

    # 1. 画圆心轨迹
    for i in range(steps_taken + 1):
        S_hist = np.array([A[0] + i * step_size, 0])
        O_hist = (S_hist + B) / 2.0
        ax.plot(O_hist[0], O_hist[1], marker='+', color='green', markersize=8, markeredgewidth=1.5, alpha=0.6)

    # 2. 画绿圆和虚轴
    if t > 0.5 and d_SB < r_wave * 2: 
        r_g_curr = np.sqrt(r_wave**2 - (d_SB/2)**2)
        ax.add_patch(plt.Circle(O_curr, r_g_curr, color='#2e8b57', fill=False, lw=2.5, linestyle='--'))
        ax.text(O_curr[0]-0.4, O_curr[1]+0.2, 'O (Center)', color='green', fontsize=11, fontweight='bold')

        # 避障计算 (找 M)
        v_obs = obstacle - O_curr
        dist_obs = np.linalg.norm(v_obs)
        theta = t * 1.2 + 0.5 
        if dist_obs > 0:
            alpha = np.arcsin(obstacle_radius / dist_obs)
            beta = np.arctan2(v_obs[1], v_obs[0])
            start_angle = np.degrees(beta - alpha)
            end_angle = np.degrees(beta + alpha)
            ax.add_patch(Arc(O_curr, r_g_curr*2, r_g_curr*2, theta1=start_angle, theta2=end_angle, 
                      color='blue', lw=4, alpha=0.8))
            if np.sin(theta - beta) > 0:
                theta = beta + alpha + 0.2
            else:
                theta = beta - alpha - 0.2

        # 算出目标 M 点的坐标
        M_x = O_curr[0] + r_g_curr * np.cos(theta)
        M_y = O_curr[1] + r_g_curr * np.sin(theta)
        
        # 画虚轴
        ax.plot([O_curr[0], M_x], [O_curr[1], M_y], color='purple', linestyle='-', lw=2, alpha=0.7)

        # ===========================================
        # 【核心修改点：消除 S 和 M 并存的视觉误导】
        # ===========================================
        
        # 图里**不再单独画一个紫色的 M 点**，而是用“跳跃痕迹”代替：
        
        # 1. 画一个“刚刚跳过来”的轨迹线（虚线），指明 S 是从上一个位置跨越到这里的
        prev_S_x = S_curr[0] - step_size
        if prev_S_x >= A[0]:
            ax.plot([prev_S_x, S_curr[0]], [0, 0], color='orange', linestyle=':', linewidth=3, alpha=0.6)
            # 在起点放个小脚印
            ax.scatter(prev_S_x, 0, c='gray', s=80, marker='x', alpha=0.4)

        # 2. S 本身放大，作为“落地”的高亮
        # (黑边 + 橙身 + 绿边，代表它是刚刚从虚轴上的 M 点踩下来的)
        ax.scatter(S_curr[0], S_curr[1], c='orange', s=300, edgecolors='black', linewidths=4, zorder=10)
        ax.scatter(S_curr[0], S_curr[1], facecolors='none', edgecolors='green', s=400, linewidths=2.5, zorder=9)
        ax.text(S_curr[0]+0.4, S_curr[1]-0.3, 'S (Landing on M)', color='black', fontsize=12, fontweight='bold')

        # 3. 走过去的脚印轨迹
        x_trail = [A[0] + i * step_size for i in range(steps_taken + 1)]
        y_trail = [0] * len(x_trail)
        ax.plot(x_trail, y_trail, color='gray', linestyle=':', lw=1.5, alpha=0.5)
        
        ax.legend(loc='upper left', fontsize=11)

    return []

ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=80)
plt.show()
