import numpy as np, math

# ────────── grid & parameters ──────────
h = k = 0.1 * math.pi
x_vals = np.arange(0, math.pi + 1e-9, h)
y_vals = np.arange(0, math.pi / 2 + 1e-9, k)
nx, ny = len(x_vals), len(y_vals)

u = np.zeros((ny, nx))

# ────────── boundary conditions ──────────
u[:, 0]  =  np.cos(y_vals)        # u(0,y)   =  cos y
u[:, -1] = -np.cos(y_vals)       # u(π,y)   = −cos y
u[0, :]  =  np.cos(x_vals)       # u(x,0)   =  cos x
u[-1, :] =  0.0                  # u(x,π/2) = 0

# ────────── source term f(x,y) = x·y ──────────
f = np.outer(y_vals, x_vals)

# ────────── Gauss–Seidel iteration ──────────
tol, max_iter = 1e-6, 10_000
for _ in range(max_iter):
    max_diff = 0.0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u_new = 0.25 * (
                u[j, i+1] + u[j, i-1] +
                u[j+1, i] + u[j-1, i] -
                h**2 * f[j, i]
            )
            max_diff = max(max_diff, abs(u_new - u[j, i]))
            u[j, i] = u_new
    if max_diff < tol:
        break

# ────────── pretty print table ──────────
col_w = 12  # fixed column width

# header
header = ["y\\x"] + [f"{x/math.pi:0.2f}π" for x in x_vals]
print("".join(f"{h:<{col_w}}" for h in header))

# rows
for j, y in enumerate(y_vals):
    row = [f"{y/math.pi:0.2f}π"] + [f"{u[j, i]: .5f}" for i in range(nx)]
    print("".join(f"{cell:<{col_w}}" for cell in row))
