import numpy as np
import matplotlib.pyplot as plt

# 定義 u1' 和 u2' 的系統
def system(t, u):
    u1, u2 = u
    du1_dt = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2_dt = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1_dt, du2_dt])

# 真實解
def exact_u1(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

# RK4方法
def runge_kutta_4(system, t0, u0, h, t_end):
    t_values = np.arange(t0, t_end+h, h)
    u_values = np.zeros((len(t_values), len(u0)))
    u_values[0] = u0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        u = u_values[i-1]
        k1 = system(t, u)
        k2 = system(t + h/2, u + h/2 * k1)
        k3 = system(t + h/2, u + h/2 * k2)
        k4 = system(t + h, u + h * k3)
        u_values[i] = u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return t_values, u_values

# 初始條件
t0 = 0
u0 = [4/3, 2/3]
t_end = 1

# 用 h=0.05 和 h=0.1 分別計算
h_list = [0.05, 0.1]

for h in h_list:
    t_values, u_values = runge_kutta_4(system, t0, u0, h, t_end)
    
    u1_exact = exact_u1(t_values)
    u2_exact = exact_u2(t_values)
    
    # 計算誤差
    error_u1 = np.abs(u1_exact - u_values[:,0])
    error_u2 = np.abs(u2_exact - u_values[:,1])

    # 顯示
    import pandas as pd
    df = pd.DataFrame({
        "t": t_values,
        "u1 (RK4)": u_values[:,0],
        "u1 (exact)": u1_exact,
        "u1 error": error_u1,
        "u2 (RK4)": u_values[:,1],
        "u2 (exact)": u2_exact,
        "u2 error": error_u2
    })
    print(f"\n結果 for h = {h}:")
    print(df)

    # 畫圖
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(t_values, u_values[:,0], 'o-', label='u1 RK4')
    plt.plot(t_values, u1_exact, 's--', label='u1 exact')
    plt.title(f"u1 vs t (h={h})")
    plt.xlabel('t')
    plt.ylabel('u1')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(t_values, u_values[:,1], 'o-', label='u2 RK4')
    plt.plot(t_values, u2_exact, 's--', label='u2 exact')
    plt.title(f"u2 vs t (h={h})")
    plt.xlabel('t')
    plt.ylabel('u2')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
