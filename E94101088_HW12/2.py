import numpy as np

# ───── 參數設定 ─────
K = 0.1
alpha = 1 / (4 * K)  # = 2.5
dr = 0.1
dt = 0.5
r_min, r_max = 0.5, 1.0
t_max = 10.0

r_vals = np.arange(r_min, r_max + 1e-12, dr)
t_vals = np.arange(0.0, t_max + 1e-12, dt)
nr, nt = len(r_vals), len(t_vals)

def apply_dirichlet(T_row, t):
    T_row[-1] = 100.0 + 40.0 * t

def robin_inner(T_row):
    T_row[0] = T_row[1] / (1 + 3 * dr)

def solve_pde(scheme):
    T = np.zeros((nt, nr))
    T[0, :] = 200.0 * (r_vals - 0.5)
    lam = alpha * dt / dr**2
    half = 0.5 * lam

    a = np.zeros(nr)
    b = np.zeros(nr)
    c = np.zeros(nr)

    for n in range(nt - 1):
        apply_dirichlet(T[n], t_vals[n])
        apply_dirichlet(T[n+1], t_vals[n+1])

        if scheme == "explicit":
            for i in range(1, nr - 1):
                r = r_vals[i]
                d2 = (T[n, i+1] - 2*T[n, i] + T[n, i-1]) / dr**2
                d1 = (T[n, i+1] - T[n, i-1]) / (2 * dr) / r
                T[n+1, i] = T[n, i] + dt * alpha * (d2 + d1)
            robin_inner(T[n+1])

        else:
            rhs = np.zeros(nr)
            for i in range(1, nr - 1):
                r = r_vals[i]
                if scheme == "implicit":
                    a[i] = -lam * (1 - dr / (2 * r))
                    b[i] = 1 + 2 * lam
                    c[i] = -lam * (1 + dr / (2 * r))
                    rhs[i] = T[n, i]
                elif scheme == "crank":
                    a[i] = -half * (1 - dr / (2 * r))
                    b[i] = 1 + lam
                    c[i] = -half * (1 + dr / (2 * r))
                    d2 = (T[n, i+1] - 2*T[n, i] + T[n, i-1]) / dr**2
                    d1 = (T[n, i+1] - T[n, i-1]) / (2 * dr) / r
                    rhs[i] = T[n, i] + half * dt * (d2 + d1)

            # Robin BC at r=0.5
            lam_fac = 1 / (1 + 3 * dr)
            a[0] = 0.0
            b[0] = 1.0
            c[0] = -lam_fac
            rhs[0] = 0.0

            # Dirichlet BC at r=1
            a[-1] = c[-1] = 0.0
            b[-1] = 1.0
            rhs[-1] = T[n+1, -1]

            # Thomas algorithm
            for i in range(1, nr):
                m = a[i] / b[i-1]
                b[i] -= m * c[i-1]
                rhs[i] -= m * rhs[i-1]
            T[n+1, -1] = rhs[-1] / b[-1]
            for i in range(nr-2, -1, -1):
                T[n+1, i] = (rhs[i] - c[i] * T[n+1, i+1]) / b[i]

    return T

# ───── 解出三種方法的結果 ─────
T_explicit = solve_pde("explicit")   # a. 前向差分
T_implicit = solve_pde("implicit")   # b. 後向差分
T_crank    = solve_pde("crank")      # c. Crank–Nicolson

# ───── 印出結果 ─────
def print_T(T, method_name):
    print(f"\n===== {method_name} method =====")
    
    col_w = 12  # 固定欄寬
    # 印欄首
    print(f"{'t\\r':>{col_w}}", end="")
    for r in r_vals:
        print(f"{r:>{col_w}.2f}", end="")
    print()
    
    # 印每個時間層
    for n, t in enumerate(t_vals):
        print(f"{t:>{col_w}.2f}", end="")
        for val in T[n]:
            print(f"{val:>{col_w}.4e}", end="")
        print()


# ───── 顯示三種結果 ─────
print_T(T_explicit, "Explicit (FTCS)")     # a
print_T(T_implicit, "Implicit (BTCS)")     # b
print_T(T_crank,    "Crank-Nicolson")      # c
