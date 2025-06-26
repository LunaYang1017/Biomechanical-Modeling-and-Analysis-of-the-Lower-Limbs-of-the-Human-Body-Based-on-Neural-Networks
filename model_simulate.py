import numpy as np
import sympy as sp
from sympy import sin, cos, diff, Matrix, symbols, simplify
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import sin, cos, diff, Matrix, symbols, lambdify

# 定义符号变量
g = symbols('g')  # 重力加速度
g_value = 9.81  # 重力加速度 (m/s^2)
# 广义坐标
yp, zp = symbols('yp zp')
theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1 theta2 theta3 theta4 theta5 theta6')
# 广义速度
dyp, dzp = symbols('dyp dzp')
dtheta1, dtheta2, dtheta3, dtheta4, dtheta5, dtheta6 = symbols('dtheta1 dtheta2 dtheta3 dtheta4 dtheta5 dtheta6')
# 广义加速度
ddyp, ddzp = symbols('ddyp ddzp')
ddtheta1, ddtheta2, ddtheta3, ddtheta4, ddtheta5, ddtheta6 = symbols(
    'ddtheta1 ddtheta2 ddtheta3 ddtheta4 ddtheta5 ddtheta6')

# 系统参数
m_pelvis = 10  # 骨盆质量(kg)
m_thigh = 7.02  # 大腿质量(kg)
m_shank = 2.44  # 小腿质量(kg)
m_foot = 1.18  # 足质量(kg)
L_pelvis = 0.2  # 骨盆宽度(m)
L_thigh = 0.4  # 大腿长度(m)
L_shank = 0.37  # 小腿长度(m)
L_foot = 0.17  # 足长度(m)

# 计算转动惯量 (均匀细杆，绕质心垂直轴)
I_thigh = (1 / 12) * m_thigh * L_thigh ** 2
I_shank = (1 / 12) * m_shank * L_shank ** 2
I_foot = (1 / 12) * m_foot * L_foot ** 2

# 关节参数 (刚度 N·m, 阻尼 N·m·s)
k_hip, c_hip = 63.66, 35
k_knee, c_knee = 79.57, 15
k_ankle, c_ankle = 33.66, 3

# ======================== 肌肉参数 ========================
F_max_hip = 120  # 髋关节最大肌肉力 (N)
F_max_knee = 150  # 膝关节最大肌肉力
F_max_ankle = 100  # 踝关节最大肌肉力

l_opt_hip = 0.3  # 髋关节最优肌纤维长度 (m)
l_opt_knee = 0.25
l_opt_ankle = 0.15

v_max_hip = 12.0  # 最大收缩速度 (m/s)
v_max_knee = 10.0
v_max_ankle = 8.0

tau_activation = 0.02  # 肌肉激活时间常数 (s)
tau_deactivation = 0.04  # 肌肉失活时间常数


# ======================== 运动学计算 ========================
def calc_kinematics():
    # 骨盆质心 (直接使用广义坐标)
    pelvis_y, pelvis_z = yp, zp

    # 左大腿质心
    thighL_y = yp + (L_thigh / 2) * sin(theta1)
    thighL_z = zp - (L_thigh / 2) * cos(theta1)

    # 左膝关节位置
    kneeL_y = yp + L_thigh * sin(theta1)
    kneeL_z = zp - L_thigh * cos(theta1)

    # 左小腿质心 (绝对角度: thigh_angle - knee_angle)
    shankL_angle = theta1 - theta3
    shankL_y = kneeL_y + (L_shank / 2) * sin(shankL_angle)
    shankL_z = kneeL_z - (L_shank / 2) * cos(shankL_angle)

    # 左踝关节位置
    ankleL_y = kneeL_y + L_shank * sin(shankL_angle)
    ankleL_z = kneeL_z - L_shank * cos(shankL_angle)

    # 左足质心 (绝对角度: shank_angle + (ankle_angle - 90°))
    footL_angle = shankL_angle + (theta5 - sp.pi / 2)
    footL_y = ankleL_y + (L_foot / 2) * sin(footL_angle)
    footL_z = ankleL_z - (L_foot / 2) * cos(footL_angle)

    # 右大腿质心 (镜像对称)
    thighR_y = yp + (L_thigh / 2) * sin(theta2)
    thighR_z = zp - (L_thigh / 2) * cos(theta2)

    # 右膝关节位置
    kneeR_y = yp + L_thigh * sin(theta2)
    kneeR_z = zp - L_thigh * cos(theta2)

    # 右小腿质心
    shankR_angle = theta2 - theta4
    shankR_y = kneeR_y + (L_shank / 2) * sin(shankR_angle)
    shankR_z = kneeR_z - (L_shank / 2) * cos(shankR_angle)

    # 右踝关节位置
    ankleR_y = kneeR_y + L_shank * sin(shankR_angle)
    ankleR_z = kneeR_z - L_shank * cos(shankR_angle)

    # 右足质心
    footR_angle = shankR_angle + (theta6 - sp.pi / 2)
    footR_y = ankleR_y + (L_foot / 2) * sin(footR_angle)
    footR_z = ankleR_z - (L_foot / 2) * cos(footR_angle)

    return {
        'pelvis': (pelvis_y, pelvis_z, 0),  # (y, z, angle)
        'thighL': (thighL_y, thighL_z, theta1),
        'shankL': (shankL_y, shankL_z, shankL_angle),
        'footL': (footL_y, footL_z, footL_angle),
        'thighR': (thighR_y, thighR_z, theta2),
        'shankR': (shankR_y, shankR_z, shankR_angle),
        'footR': (footR_y, footR_z, footR_angle)
    }


# ======================== 能量计算 ========================
def calc_energy(kinematics):
    segments = [
        ('pelvis', m_pelvis, 0, kinematics['pelvis']),
        ('thighL', m_thigh, I_thigh, kinematics['thighL']),
        ('shankL', m_shank, I_shank, kinematics['shankL']),
        ('footL', m_foot, I_foot, kinematics['footL']),
        ('thighR', m_thigh, I_thigh, kinematics['thighR']),
        ('shankR', m_shank, I_shank, kinematics['shankR']),
        ('footR', m_foot, I_foot, kinematics['footR'])
    ]

    T = 0  # 总动能
    V = 0  # 总势能

    # 广义坐标和时间导数
    q = [yp, zp, theta1, theta2, theta3, theta4, theta5, theta6]
    dq = [dyp, dzp, dtheta1, dtheta2, dtheta3, dtheta4, dtheta5, dtheta6]

    for name, m, I, (y_sym, z_sym, angle_sym) in segments:
        # 计算速度 (dy/dt, dz/dt)
        dy = diff(y_sym, yp) * dyp + diff(y_sym, zp) * dzp + \
             diff(y_sym, theta1) * dtheta1 + diff(y_sym, theta2) * dtheta2 + \
             diff(y_sym, theta3) * dtheta3 + diff(y_sym, theta4) * dtheta4 + \
             diff(y_sym, theta5) * dtheta5 + diff(y_sym, theta6) * dtheta6
        dz = diff(z_sym, yp) * dyp + diff(z_sym, zp) * dzp + \
             diff(z_sym, theta1) * dtheta1 + diff(z_sym, theta2) * dtheta2 + \
             diff(z_sym, theta3) * dtheta3 + diff(z_sym, theta4) * dtheta4 + \
             diff(z_sym, theta5) * dtheta5 + diff(z_sym, theta6) * dtheta6

        # 平动动能
        T_trans = 0.5 * m * (dy ** 2 + dz ** 2)

        # 转动动能 (角速度 = d(angle)/dt)
        d_angle = diff(angle_sym, yp) * dyp + diff(angle_sym, zp) * dzp + \
                  diff(angle_sym, theta1) * dtheta1 + diff(angle_sym, theta2) * dtheta2 + \
                  diff(angle_sym, theta3) * dtheta3 + diff(angle_sym, theta4) * dtheta4 + \
                  diff(angle_sym, theta5) * dtheta5 + diff(angle_sym, theta6) * dtheta6
        T_rot = 0.5 * I * d_angle ** 2

        T += T_trans + T_rot
        V += m * g * z_sym  # 重力势能

    return T, V, q, dq


# ======================== 动力学方程 ========================
def lagrange_equations(T, V, q, dq):
    L = T - V
    n = len(q)
    eqns = []

    # 广义力 (关节被动力)
    Q = Matrix([
        0,  # yp
        0,  # zp
        -k_hip * theta1 - c_hip * dtheta1,  # 左髋
        -k_hip * theta2 - c_hip * dtheta2,  # 右髋
        -k_knee * theta3 - c_knee * dtheta3,  # 左膝
        -k_knee * theta4 - c_knee * dtheta4,  # 右膝
        -k_ankle * (theta5 - sp.pi / 2) - c_ankle * dtheta5,  # 左踝
        -k_ankle * (theta6 - sp.pi / 2) - c_ankle * dtheta6  # 右踝
    ])

    # 构建拉格朗日方程
    for i in range(n):
        dL_dqdot_i = diff(L, dq[i])
        dL_dq_i = diff(L, q[i])

        # d/dt(dL/dqdot_i)
        d_dt = 0
        for j in range(n):
            d_dt += diff(dL_dqdot_i, q[j]) * dq[j] + diff(dL_dqdot_i, dq[j]) * symbols(f'dd{q[j]}')

        eqn = d_dt - dL_dq_i - Q[i]
        eqns.append(eqn)

    return eqns


# ======================== 数值模拟 ========================
def simulate_dynamics(initial_state, target_state, t_span, t_eval):
    """
    执行瞬态响应模拟
    :param initial_state: 初始状态 [yp, zp, theta1-6, dyp, dzp, dtheta1-6] (16个元素)
    :param target_state: 目标状态 [yp, zp, theta1-6] (8个元素)
    :param t_span: 时间范围 (start, end)
    :param t_eval: 评估时间点
    :return: 模拟结果
    """
    # 计算运动学
    kinematics = calc_kinematics()

    # 计算能量
    T, V, q, dq = calc_energy(kinematics)

    # 推导拉格朗日方程
    eqns = lagrange_equations(T, V, q, dq)

    # 提取质量矩阵M和力向量F
    n = len(q)
    M = sp.zeros(n, n)
    F = sp.zeros(n, 1)

    # 广义加速度
    ddq_sym = [ddyp, ddzp, ddtheta1, ddtheta2, ddtheta3, ddtheta4, ddtheta5, ddtheta6]

    for i, eq in enumerate(eqns):
        # 提取广义加速度的系数
        for j in range(n):
            M[i, j] = diff(eq, ddq_sym[j])

        # 剩余部分构成F
        F[i] = -eq.subs({sym: 0 for sym in ddq_sym})

    # 用实际重力加速度替换符号
    M = M.subs(g, g_value)
    F = F.subs(g, g_value)

    # 创建符号变量列表
    all_syms = q + dq

    # 将符号表达式转换为数值函数
    M_func = lambdify(all_syms, M, 'numpy')
    F_func = lambdify(all_syms, F, 'numpy')

    def get_muscle_activation(t, joint_index):
        """预定义的肌肉激活模式函数"""
        # 不同关节有不同的激活模式
        if joint_index in [0, 1]:  # 髋关节
            # 早期激活，后期保持
            return min(t / 0.6, 0.9) if t < 1.0 else max(0.5 - (t - 1.0) * 0.3, 0.6)

        elif joint_index in [2, 3]:  # 膝关节
            # 中期激活
            return 0 if t < 0.3 else min((t - 0.3) / 0.8, 0.6)

        elif joint_index in [4, 5]:  # 踝关节
            # 后期激活
            return 0 if t < 0.6 else min((t - 0.6) / 0.8, 0.5)

    # 定义微分方程
    def dynamics(t, y):
        """
        状态导数的动力学方程
        :param t: 时间
        :param y: 状态向量 [q0-7, dq0-7]
        :return: dy/dt
        """
        # 拆分状态向量
        q_vals = y[:8]  # 位置
        dq_vals = y[8:]  # 速度

        # # 添加目标状态的影响 (PD控制)
        # control_gain = 20.0
        # damping_gain = 5.0
        # control_torques = np.zeros(8)
        #
        # # 计算控制力矩 (PD控制器)
        # for i in range(2, 8):  # 从theta1开始
        #     error = q_vals[i] - target_state[i]
        #     derror = dq_vals[i]
        #     control_torques[i] = -control_gain * error - damping_gain * derror

        muscle_torques = np.zeros(8)
        F_max = [F_max_hip, F_max_hip, F_max_knee, F_max_knee, F_max_ankle, F_max_ankle]

        # for i in range(6):
        #     activation = get_muscle_activation(t, i)
        #     # 简单比例控制
        #     muscle_torques[i + 2] = activation * F_max[i]

        # 计算肌肉力矩（修正方向）
        for i in range(6):
            activation = get_muscle_activation(t, i)
            if i in [0, 1]:  # 髋关节 - 产生伸展力矩
                muscle_torques[i + 2] = -activation * F_max_hip * (q_vals[i + 2] - target_state[i + 2])
            elif i in [2, 3]:  # 膝关节 - 产生伸展力矩
                muscle_torques[i + 2] = activation * F_max_knee * (q_vals[i + 2] - target_state[i + 2])
            elif i in [4, 5]:  # 踝关节 - 产生跖屈力矩
                muscle_torques[i + 2] = -activation * F_max_ankle * (q_vals[i + 2] - target_state[i + 2])

        # 计算M和F矩阵
        M_val = M_func(*q_vals, *dq_vals)
        F_val = F_func(*q_vals, *dq_vals)

        # 应用控制力矩
        F_val = F_val.flatten() + muscle_torques

        # 求解加速度
        try:
            ddq_vals = np.linalg.solve(M_val, F_val)
        except np.linalg.LinAlgError:
            # 处理奇异矩阵情况
            ddq_vals = np.zeros(8)

        # 组装状态导数 [速度, 加速度]
        dydt = np.concatenate((dq_vals, ddq_vals))

        return dydt

    # 执行模拟
    sol = solve_ivp(dynamics, t_span, initial_state, t_eval=t_eval, method='RK45')

    return sol


# ======================== 可视化结果 ========================
# def plot_results(t, y, target_state):
#     """
#     绘制模拟结果
#     :param t: 时间数组
#     :param y: 状态矩阵 (16 x n)
#     :param target_state: 目标状态
#     """
#     plt.figure(figsize=(15, 12))
#
#     # 骨盆位置
#     # plt.subplot(3, 1, 1)
#     # plt.plot(t, y[0], 'b-', label='y position')
#     # plt.plot(t, y[1], 'r-', label='z position')
#     # plt.axhline(y=target_state[0], color='b', linestyle='--', label='Target y')
#     # plt.axhline(y=target_state[1], color='r', linestyle='--', label='Target z')
#     # plt.xlabel('Time (s)')
#     # plt.ylabel('Position (m)')
#     # plt.title('Pelvis Position')
#     # plt.legend()
#     # plt.grid(True)
#
#
#     # 关节角度
#     plt.subplot(2, 1, 1)
#     joints = ['Hip L', 'Hip R', 'Knee L', 'Knee R', 'Ankle L', 'Ankle R']
#     colors = ['b', 'g', 'r', 'c', 'm', 'y']
#     for i in range(6):
#         plt.plot(t, np.rad2deg(y[2 + i]), colors[i], label=joints[i])
#         plt.axhline(y=np.rad2deg(target_state[2 + i]), color=colors[i], linestyle='--')
#
#     plt.xlabel('Time (s)')
#     plt.ylabel('Joint Angle (deg)')
#     plt.title('Joint Angles')
#     plt.legend()
#     plt.grid(True)
#
#     # 关节角速度
#     plt.subplot(2, 1, 2)
#     for i in range(6):
#         plt.plot(t, np.rad2deg(y[8 + 2 + i]), colors[i], label=joints[i] + ' velocity')
#
#     plt.xlabel('Time (s)')
#     plt.ylabel('Joint Velocity (deg/s)')
#     plt.title('Joint Angular Velocities')
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig('joint_response.png')
#     plt.show()

# # ======================== 可视化结果 (修正版) ========================
# def plot_results(t, y, target_state):
#     """
#     绘制模拟结果（使用线型区分而非颜色）
#     :param t: 时间数组
#     :param y: 状态矩阵 (16 x n)
#     :param target_state: 目标状态
#     """
#     plt.figure(figsize=(15, 12))
#
#     # 为不同关节类型设置不同的线型
#     joint_styles = {
#         'Hip': {'linestyle': '-', 'linewidth': 2, 'color': 'black'},  # 实线表示髋关节
#         'Knee': {'linestyle': '--', 'linewidth': 1.5, 'color': 'black'},  # 虚线表示膝关节
#         'Ankle': {'linestyle': ':', 'linewidth': 1.5, 'color': 'black'}  # 点线表示踝关节
#     }
#
#     # 关节角度
#     plt.subplot(2, 1, 1)
#
#     # 左腿关节 - 使用标准线型
#     # plt.plot(t, np.rad2deg(y[2]), label='Hip L', **{**joint_styles['Hip'], 'dashes': (None, None)})
#     # plt.plot(t, np.rad2deg(y[4]), label='Knee L', **{**joint_styles['Knee'], 'dashes': (None, None)})
#     # plt.plot(t, np.rad2deg(y[6]), label='Ankle L', **{**joint_styles['Ankle'], 'dashes': (None, None)})
#
#     # 右腿关节 - 使用虚线样式区分
#     plt.plot(t, np.rad2deg(y[3]), label='Hip ', **{**joint_styles['Hip'], 'dashes': (6, 2)})
#     plt.plot(t, np.rad2deg(y[5]), label='Knee ', **{**joint_styles['Knee'], 'dashes': (3, 2)})
#     plt.plot(t, np.rad2deg(y[7]), label='Ankle ', **{**joint_styles['Ankle'], 'dashes': (2, 2)})
#
#     # 目标角度线（灰色虚线）
#     plt.axhline(y=np.rad2deg(target_state[2]), color='gray', linestyle='--', alpha=0.5)
#     plt.axhline(y=np.rad2deg(target_state[3]), color='gray', linestyle='--', alpha=0.5)
#     plt.axhline(y=np.rad2deg(target_state[4]), color='gray', linestyle='--', alpha=0.5)
#     plt.axhline(y=np.rad2deg(target_state[5]), color='gray', linestyle='--', alpha=0.5)
#     plt.axhline(y=np.rad2deg(target_state[6]), color='gray', linestyle='--', alpha=0.5)
#     plt.axhline(y=np.rad2deg(target_state[7]), color='gray', linestyle='--', alpha=0.5)
#
#     plt.xlabel('Time (s)')
#     plt.ylabel('Joint Angle (deg)')
#     plt.title('Joint Angles')
#     plt.legend(ncol=2)
#     plt.grid(True, linestyle=':', alpha=0.7)
#
#     # 关节角速度
#     plt.subplot(2, 1, 2)
#
#     # 左腿关节 - 使用标准线型
#     # plt.plot(t, np.rad2deg(y[10]), label='Hip L velocity', **{**joint_styles['Hip'], 'dashes': (None, None)})
#     # plt.plot(t, np.rad2deg(y[12]), label='Knee L velocity', **{**joint_styles['Knee'], 'dashes': (None, None)})
#     # plt.plot(t, np.rad2deg(y[14]), label='Ankle L velocity', **{**joint_styles['Ankle'], 'dashes': (None, None)})
#
#     # 右腿关节 - 使用虚线样式区分
#     plt.plot(t, np.rad2deg(y[11]), label='Hip  velocity', **{**joint_styles['Hip'], 'dashes': (6, 2)})
#     plt.plot(t, np.rad2deg(y[13]), label='Knee  velocity', **{**joint_styles['Knee'], 'dashes': (3, 2)})
#     plt.plot(t, np.rad2deg(y[15]), label='Ankle  velocity', **{**joint_styles['Ankle'], 'dashes': (2, 2)})
#
#     plt.xlabel('Time (s)')
#     plt.ylabel('Joint Velocity (deg/s)')
#     plt.title('Joint Angular Velocities')
#     plt.legend(ncol=2)
#     plt.grid(True, linestyle=':', alpha=0.7)
#
#     plt.tight_layout()
#     plt.savefig('joint_response_bw.png', dpi=300)
#     plt.show()

def plot_results(t, y, target_state):
    """
    绘制模拟结果（使用线型区分而非颜色）
    :param t: 时间数组
    :param y: 状态矩阵 (16 x n)
    :param target_state: 目标状态
    """
    # ---------------------------- 全局字体设置（关键修改） ----------------------------
    # 设置全局字体大小（6号以上，12磅≈小五号，14磅≈四号）
    plt.rcParams.update({
        'font.size': 20,          # 默认基础字体大小（覆盖大部分文本）
        'axes.labelsize': 20,     # 坐标轴标签（x/y轴名称）字号
        'axes.titlesize': 24,     # 坐标轴标题（如"Joint Angles"）字号
        'xtick.labelsize': 20,    # x轴刻度标签字号
        'ytick.labelsize': 20,    # y轴刻度标签字号
        'legend.fontsize': 18,    # 图例字号
        'figure.titlesize': 18    # 图整体标题字号（若有）
    })
    # --------------------------------------------------------------------------------------

    plt.figure(figsize=(15, 12))

    # 为不同关节类型设置不同的线型
    joint_styles = {
        'Hip': {'linestyle': '-', 'linewidth': 2, 'color': 'black'},  # 实线表示髋关节
        'Knee': {'linestyle': '--', 'linewidth': 1.5, 'color': 'black'},  # 虚线表示膝关节
        'Ankle': {'linestyle': ':', 'linewidth': 1.5, 'color': 'black'}  # 点线表示踝关节
    }

    # 关节角度
    plt.subplot(2, 1, 1)

    # 右腿关节 - 使用虚线样式区分
    plt.plot(t, np.rad2deg(y[3]), label='Hip ', **{**joint_styles['Hip'], 'dashes': (6, 2)})
    plt.plot(t, np.rad2deg(y[5]), label='Knee ', **{**joint_styles['Knee'], 'dashes': (3, 2)})
    plt.plot(t, np.rad2deg(y[7]), label='Ankle ', **{**joint_styles['Ankle'], 'dashes': (2, 2)})

    # 目标角度线（灰色虚线）
    plt.axhline(y=np.rad2deg(target_state[2]), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=np.rad2deg(target_state[3]), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=np.rad2deg(target_state[4]), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=np.rad2deg(target_state[5]), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=np.rad2deg(target_state[6]), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=np.rad2deg(target_state[7]), color='gray', linestyle='--', alpha=0.5)

    # ---------------------------- 显式指定关键文本字号 ----------------------------
    plt.xlabel('Time (s)', fontsize=24)       # x轴标签字号（14磅）
    plt.ylabel('Joint Angle (deg)', fontsize=24)  # y轴标签字号（14磅）
    plt.title('Joint Angles', fontsize=30)    # 子图标题字号（16磅）
    plt.legend(ncol=2, fontsize=22)           # 图例字号（12磅）
    plt.grid(True, linestyle=':', alpha=0.7)

    # 关节角速度
    plt.subplot(2, 1, 2)

    # 右腿关节 - 使用虚线样式区分
    plt.plot(t, np.rad2deg(y[11]), label='Hip  velocity', **{**joint_styles['Hip'], 'dashes': (6, 2)})
    plt.plot(t, np.rad2deg(y[13]), label='Knee  velocity', **{**joint_styles['Knee'], 'dashes': (3, 2)})
    plt.plot(t, np.rad2deg(y[15]), label='Ankle  velocity', **{**joint_styles['Ankle'], 'dashes': (2, 2)})

    # ---------------------------- 显式指定关键文本字号 ----------------------------
    plt.xlabel('Time (s)', fontsize=24)       # x轴标签字号（14磅）
    plt.ylabel('Joint Velocity (deg/s)', fontsize=24)  # y轴标签字号（14磅）
    plt.title('Joint Angular Velocities', fontsize=30)  # 子图标题字号（16磅）
    plt.legend(ncol=2, fontsize=22)           # 图例字号（12磅）
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('joint_response_bw.png', dpi=300)
    plt.show()

# ======================== 主程序 ========================
if __name__ == "__main__":
    # 计算运动学
    kinematics = calc_kinematics()

    # 计算能量
    T, V, q, dq = calc_energy(kinematics)
    print("动能 T 和势能 V 已计算")

    # 推导拉格朗日方程
    print("推导动力学方程... (可能需要较长时间)")
    eqns = lagrange_equations(T, V, q, dq)

    # 提取质量矩阵M和力向量F
    n = len(q)
    M = sp.zeros(n, n)
    F = sp.zeros(n, 1)

    # 广义加速度
    ddq_sym = [ddyp, ddzp, ddtheta1, ddtheta2, ddtheta3, ddtheta4, ddtheta5, ddtheta6]

    for i, eq in enumerate(eqns):
        # 提取广义加速度的系数
        for j in range(n):
            M[i, j] = diff(eq, ddq_sym[j])

        # 剩余部分构成F
        F[i] = -eq.subs({sym: 0 for sym in ddq_sym})

    print("质量矩阵 M 和力向量 F 已提取")

    # 简化并输出部分结果 (由于表达式复杂，仅展示M[0,0]和F[0])
    print("\n示例：M[0,0] =")
    print(simplify(M[0, 0]))

    print("\n示例：F[0] =")
    print(simplify(F[0]))

    # 保存到文件 (完整矩阵通常很大)
    with open("dynamics_equations.txt", "w") as f:
        f.write("质量矩阵 M:\n")
        f.write(str(M))
        f.write("\n\n力向量 F:\n")
        f.write(str(F))

    print("\n动力学方程已保存至 dynamics_equations.txt")

    # 定义初始状态 (半蹲姿势)
    # 初始位置
    yp0 = 0.0  # 骨盆y位置 (m)
    zp0 = 0.3  # 骨盆z位置 (m) - 半蹲时较低

    # 初始关节角度 (弧度)
    # 髋关节: 屈曲约45度 (-45°)
    # 膝关节: 屈曲约45度 (45°)
    # 踝关节: 背屈约15度 (90° - 15° = 75°)
    theta10 = -np.deg2rad(60)  # 左髋
    theta20 = -np.deg2rad(60)  # 右髋
    theta30 = np.deg2rad(85)  # 左膝
    theta40 = np.deg2rad(85)  # 右膝
    theta50 = np.deg2rad(75)  # 左踝 (90° - 背屈15°)
    theta60 = np.deg2rad(75)  # 右踝

    # 初始速度 (全部为零)
    dyp0, dzp0 = 0.0, 0.0
    dtheta10, dtheta20, dtheta30, dtheta40, dtheta50, dtheta60 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 目标状态 (直立姿势)
    yp_target = 0.0  # 骨盆y位置 (m)
    zp_target = 1.0  # 骨盆z位置 (m) - 直立时较高
    theta1_target = 0.0  # 左髋
    theta2_target = 0.0  # 右髋
    theta3_target = 0.0  # 左膝
    theta4_target = 0.0  # 右膝
    theta5_target = np.deg2rad(90)  # 左踝 (90°)
    theta6_target = np.deg2rad(90)  # 右踝

    # 初始状态向量 (位置 + 速度)
    initial_state = [
        yp0, zp0,
        theta10, theta20, theta30, theta40, theta50, theta60,
        dyp0, dzp0,
        dtheta10, dtheta20, dtheta30, dtheta40, dtheta50, dtheta60
    ]

    # 目标状态向量 (仅位置)
    target_state = [
        yp_target, zp_target,
        theta1_target, theta2_target, theta3_target, theta4_target,
        theta5_target, theta6_target
    ]

    # 时间参数
    t_span = (0, 2.0)  # 模拟时间范围 (0到2秒)
    t_eval = np.linspace(0, 2.0, 500)  # 评估时间点

    print("开始瞬态响应模拟...")
    sol = simulate_dynamics(initial_state, target_state, t_span, t_eval)
    print("模拟完成!")

    # 可视化结果
    plot_results(sol.t, sol.y, target_state)

    # 保存结果
    np.savez('simulation_results.npz',
             t=sol.t,
             y=sol.y,
             initial_state=initial_state,
             target_state=target_state)

    print("结果已保存为 simulation_results.npz")