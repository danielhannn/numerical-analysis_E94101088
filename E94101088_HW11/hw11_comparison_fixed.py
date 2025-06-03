
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import sympy as sp

# ========================================
# (a) Shooting Method
# ========================================
def ode_system(x, Y):
    y, dy = Y
    d2y = -(x + 1) * dy + 2 * y + (1 - x ** 2) * np.exp(x)
    return [dy, d2y]

def shoot(slope_guess):
    sol = solve_ivp(ode_system, [0, 1], [1, slope_guess], t_eval=[1])
    return sol.y[0, -1] - 2

def find_bracket():
    test_vals = np.linspace(-10, 10, 200)
    for i in range(len(test_vals) - 1):
        a = test_vals[i]
        b = test_vals[i + 1]
        fa = shoot(a)
        fb = shoot(b)
        if fa * fb < 0:
            return (a, b)
    raise ValueError("找不到適合的 bracket：f(a) 與 f(b) 沒有異號")

bracket = find_bracket()
res = root_scalar(shoot, bracket=bracket, method='bisect')
slope = res.root

x_vals = np.linspace(0, 1, 11)
sol = solve_ivp(ode_system, [0, 1], [1, slope], t_eval=x_vals)
y_shoot = sol.y[0]

# ========================================
# (b) Finite Difference Method
# ========================================
h = 0.1
x = np.arange(0, 1 + h, h)
n = len(x)

A = np.zeros((n, n))
b = np.zeros(n)

A[0, 0] = 1
b[0] = 1
A[-1, -1] = 1
b[-1] = 2

for i in range(1, n - 1):
    xi = x[i]
    A[i, i - 1] = 1 / h**2 - (xi + 1) / (2 * h)
    A[i, i]     = -2 / h**2 - 2
    A[i, i + 1] = 1 / h**2 + (xi + 1) / (2 * h)
    b[i] = -(1 - xi**2) * np.exp(xi)

y_fd = np.linalg.solve(A, b)

# ========================================
# (c) Variational Method (修正版 trial function)
# y(x) = 1 + x + a1 * x * (1 - x)
# ========================================
x_sym = sp.Symbol('x')
a1 = sp.Symbol('a1')
y_trial = 1 + x_sym + a1 * x_sym * (1 - x_sym)

dy = sp.diff(y_trial, x_sym)
d2y = sp.diff(dy, x_sym)
f_expr = d2y + (x_sym + 1) * dy - 2 * y_trial - (1 - x_sym ** 2) * sp.exp(x_sym)
L = f_expr ** 2
J = sp.integrate(L, (x_sym, 0, 1))
a1_val = float(sp.solve(sp.diff(J, a1), a1)[0])

f_numeric = sp.lambdify(x_sym, y_trial.subs(a1, a1_val), 'numpy')
y_ritz = f_numeric(x)

# ========================================
# 比較輸出
# ========================================
print(f"{'x':>6} | {'Shooting':>12} | {'Finite Diff':>12} | {'Variational':>12}")
print("-" * 52)
for xi, y1, y2, y3 in zip(x, y_shoot, y_fd, y_ritz):
    print(f"{xi:6.2f} | {y1:12.6f} | {y2:12.6f} | {y3:12.6f}")
