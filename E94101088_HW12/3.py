import numpy as np

# ───── 1. Grid setting ─────
Nr = 11                 # number of r points
Ntheta = 16            # number of θ points
r_min, r_max = 0.5, 1.0
theta_min, theta_max = 0, np.pi / 3

r_vals = np.linspace(r_min, r_max, Nr)
theta_vals = np.linspace(theta_min, theta_max, Ntheta)
dr = r_vals[1] - r_vals[0]
dtheta = theta_vals[1] - theta_vals[0]

# ───── 2. Initialize T(r, θ) ─────
T = np.zeros((Nr, Ntheta))

# ───── 3. Boundary conditions ─────
T[0, :]  = 50             # r = 0.5
T[-1, :] = 100            # r = 1
T[:,  0] = 0              # θ = 0
T[:, -1] = 0              # θ = π/3

# ───── 4. Gauss–Seidel iteration ─────
max_iter = 10000
tol = 1e-5

for iteration in range(max_iter):
    max_diff = 0.0
    for i in range(1, Nr - 1):
        r = r_vals[i]
        for j in range(1, Ntheta - 1):
            Trp = T[i + 1, j]
            Trm = T[i - 1, j]
            Tthp = T[i, j + 1]
            Tthm = T[i, j - 1]

            term_r = (Trp + Trm) / dr**2 + (Trp - Trm) / (2 * r * dr)
            term_th = (Tthp + Tthm) / (r**2 * dtheta**2)
            denom = 2 / dr**2 + 2 / (r**2 * dtheta**2)

            new_val = (term_r + term_th) / denom
            diff = abs(new_val - T[i, j])
            max_diff = max(max_diff, diff)
            T[i, j] = new_val

    if max_diff < tol:
        break

# ───── 5. Print T(r, θ) at final result ─────

print("\n==== T(r, θ) grid ====")
print("θ values:", ["{:.2f}".format(theta) for theta in theta_vals])
print("Each row shows T at fixed r:")

for i, r in enumerate(r_vals):
    values = "  ".join(f"{T[i, j]:6.2f}" for j in range(Ntheta))
    print(f"r = {r:.2f} → {values}")
