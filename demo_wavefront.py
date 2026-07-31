import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===========================
# 1. 环境与参数设置
# ===========================
A = np.array([-3.0, 0.0])  # 起点
B = np.array([3.0, 0.0])   # 终点
v_wave = 1.0               # 波前扩散速度
v_agent = 0.2              # 智能体(S)移动速度
frames = 100               # 动画总帧数
total_time = 10.0          # 总模拟时间

# 设置绘图画布 (上下两张图)
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10))
plt.subplots_adjust(hspace=0.3)

# 初始化上下图的元素列表
patches_top = []
patches_bot = []
trail_circles = [] # 用于记录下半图历史的绿色圆

def init():
    # ---- 上半图：波前干涉 ----
    ax_top.set_title("Top: Wavefront Interference (Start A to End B)", fontsize=12)
    ax_top.set_xlim(-6, 6)
    ax_top.set_ylim(-4, 4)
    ax_top.set_aspect('equal')
    ax_top.grid(True, linestyle=':', alpha=0.5)
    ax_top.scatter(A[0], A[1], c='black', s=50, label='Start (A)')
    ax_top.scatter(B[0], B[1], c='black', s=50, label='End (B)')
    ax_top.legend(loc='upper right')

    # ---- 下半图：绿色圆演化 ----
    ax_bot.set_title("Bottom: Green Circle Evolution as Agent Moves", fontsize=12)
    ax_bot.set_xlim(-6, 6)
    ax_bot.set_ylim(-4, 4)
    ax_bot.set_aspect('equal')
    ax_bot.grid(True, linestyle=':', alpha=0.5)
    ax_bot.scatter(A[0], A[1], c='black', s=50)
    ax_bot.text(A[0]-0.5, A[1]-0.6, 'Start (A)', fontsize=9)
    ax_bot.scatter(B[0], B[1], c='black', s=50)
    ax_bot.text(B[0]-0.5, B[1]-0.6, 'End (B)', fontsize=9)
    
    return []

# 核心更新函数，每一帧调用一次
def update(frame):
    # 时间推进
    t = (frame / frames) * total_time
    
    # 1. 智能体(S)的实时位置 (向前移动)
    S_curr = A + np.array([min(v_agent * t, 6), 0]) # 限制不能跑出视野
    
    # 当前波前半径
    r_wave = v_wave * t
    
    # ---- 绘制上半图：红蓝波 + 绿圆 ----
    ax_top.clear()
    ax_top.set_title(f"Top: t={t:.1f}s", fontsize=12)
    ax_top.set_xlim(-6, 6)
    ax_top.set_ylim(-4, 4)
    ax_top.set_aspect('equal')
    ax_top.grid(True, linestyle=':', alpha=0.5)
    ax_top.scatter(A[0], A[1], c='black', s=50, label='Start (A)')
    ax_top.scatter(B[0], B[1], c='black', s=50, label='End (B)')
    
    # 画蓝波 (A发出)
    circle_A = plt.Circle(A, r_wave, color='blue', fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
    ax_top.add_patch(circle_A)
    # 画红波 (B发出)
    circle_B = plt.Circle(B, r_wave, color='red', fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
    ax_top.add_patch(circle_B)
    # 计算并绘制当前时刻的绿色圆 (干涉圆)
    d = np.linalg.norm(B - A)
    if r_wave > d/2: # 当两波相交时
        O_top = (A + B) / 2.0
        r_g_top = np.sqrt(r_wave**2 - (d/2)**2)
        green_circle_top = plt.Circle(O_top, r_g_top, color='green', fill=False, linewidth=3)
        ax_top.add_patch(green_circle_top)
        ax_top.text(O_top[0]-1, O_top[1]+0.8, "Green Circle", color='green', fontsize=10)
    ax_top.legend(loc='upper right')

    # ---- 绘制下半图：移动的绿圆 (轨迹) ----
    ax_bot.clear()
    ax_bot.set_title(f"Bottom: t={t:.1f}s", fontsize=12)
    ax_bot.set_xlim(-6, 6)
    ax_bot.set_ylim(-4, 4)
    ax_bot.set_aspect('equal')
    ax_bot.grid(True, linestyle=':', alpha=0.5)
    ax_bot.scatter(A[0], A[1], c='black', s=50)
    ax_bot.text(A[0]-0.5, A[1]-0.6, 'Start (A)', fontsize=9)
    ax_bot.scatter(B[0], B[1], c='black', s=50)
    ax_bot.text(B[0]-0.5, B[1]-0.6, 'End (B)', fontsize=9)
    
    # 画智能体S的当前位置
    ax_bot.scatter(S_curr[0], S_curr[1], c='orange', s=60, label='Agent (S)')
    ax_bot.text(S_curr[0]+0.2, S_curr[1]+0.3, 'S', color='orange', fontsize=12)

    # 核心：计算并画出随位置变化的绿色圆
    # 原理：当前绿色圆是基于 S_curr 和 终点 B 算出来的
    if t > 0.1:
        d_t = np.linalg.norm(B - S_curr)
        # 圆心在中点
        O_curr = (S_curr + B) / 2.0
        # 公式：r_g^2 = r_wave^2 - (d_t/2)^2
        if r_wave > d_t/2:
            r_g_curr = np.sqrt(r_wave**2 - (d_t/2)**2)
            
            # 画当前绿圆 (深色粗线)
            green_circle_curr = plt.Circle(O_curr, r_g_curr, color='green', fill=False, linewidth=2.5, label='Current Green Circle')
            ax_bot.add_patch(green_circle_curr)
            
            # 加上“虚轴”和“M点” (为了演示高频旋转，让虚轴随着时间慢慢逆时针旋转)
            theta = t * 0.5  # 虚轴旋转角度
            M_x = O_curr[0] + r_g_curr * np.cos(theta)
            M_y = O_curr[1] + r_g_curr * np.sin(theta)
            
            # 画虚轴
            ax_bot.plot([O_curr[0] - r_g_curr*1.3*np.cos(theta), O_curr[0] + r_g_curr*1.3*np.cos(theta)],
                        [O_curr[1] - r_g_curr*1.3*np.sin(theta), O_curr[1] + r_g_curr*1.3*np.sin(theta)],
                        color='gray', linestyle='--', linewidth=1.5, label='Virtual Axis')
            # 画M点
            ax_bot.scatter(M_x, M_y, c='purple', s=80, zorder=5)
            ax_bot.text(M_x+0.1, M_y+0.3, 'M', color='purple', fontsize=12)
            ax_bot.legend(loc='upper right')

    return []

# ===========================
# 生成动画并保存为 GIF
# ===========================
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=80)

# 保存为 GIF (需要安装 Pillow 库，如果没有，会报错。或者直接删掉.save这行只看弹窗)
try:
    ani.save('circle_walker_demo.gif', writer='pillow', fps=15)
    print("✅ 动图已保存为 circle_walker_demo.gif")
except:
    print("⚠️ 无法保存GIF，但动画窗口正在弹出（请确保已安装 pillow: pip install pillow）")

plt.show()
