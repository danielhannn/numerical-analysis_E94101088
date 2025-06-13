import numpy as np

# ───── 1. Grid parameters ─────
dx = dt = 0.1
x_vals = np.arange(0, 1 + dx, dx)     # 0 → 1
t_vals = np.arange(0, 1 + dt, dt)     # 可依需求延長時間
nx = len(x_vals)
nt = len(t_vals)

# ───── 2. Initialize solution array ─────
p = np.zeros((nt, nx))

# Initial condition p(x,0) = cos(2πx)
p[0, :] = np.cos(2 * np.pi * x_vals)

# First time step using initial velocity ∂p/∂t = 2π sin(2πx)
# Use central time approximation:
# p₁ = p₀ + dt * ∂p/∂t + 0.5 * dt² * ∂²p/∂x²
for i in range(1, nx - 1):
    d2p_dx2 = (p[0, i+1] - 2*p[0, i] + p[0, i-1]) / dx**2
    dp_dt_0 = 2 * np.pi * np.sin(2 * np.pi * x_vals[i])
    p[1, i] = p[0, i] + dt * dp_dt_0 + 0.5 * dt**2 * d2p_dx2

# Apply boundary conditions at t₁
p[1, 0] = 1     # p(0,t) = 1
p[1, -1] = 2    # p(1,t) = 2

# ───── 3. Time stepping (explicit wave equation) ─────
for n in range(1, nt - 1):
    for i in range(1, nx - 1):
        p[n+1, i] = (2 * p[n, i] - p[n-1, i] +
                     (dt**2 / dx**2) * (p[n, i+1] - 2*p[n, i] + p[n, i-1]))
    # Boundary conditions
    p[n+1, 0]  = 1
    p[n+1, -1] = 2

# ───── 4. Display p(x, t) for all time steps ─────
col_w = 12  # 欄寬

# 印欄標：x 座標
print("\n" + " " * col_w + "".join(f"{x:>{col_w}.2f}" for x in x_vals))

# 印每個時間層
for n, t in enumerate(t_vals):
    print(f"{t:>{col_w}.2f}", end="")  # 時間欄
    for val in p[n]:
        print(f"{val:>{col_w}.4e}", end="")  # 對應 p 值
    print()
