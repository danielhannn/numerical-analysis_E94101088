import numpy as np
from scipy.optimize import root_scalar

# 方案 C 的資料點
x_vals = np.array([0.3, 0.5, 0.6])
y_vals = np.array([0.740818, 0.606531, 0.548812])

# 反轉為 y -> x 來做反插值
x_vals = x_vals[::-1]
y_vals = y_vals[::-1]

def newton_divided_diff(x_vals, y_vals):
    n = len(x_vals)
    coef = np.copy(x_vals)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (y_vals[j:n] - y_vals[j-1:n-1])
    return coef

def evaluate_newton_polynomial(coef, y_vals, y):
    n = len(coef)
    result = coef[0]
    term = 1.0
    for i in range(1, n):
        term *= (y - y_vals[i - 1])
        result += coef[i] * term
    return result

# 插值並求解 x ≈ e^{-x}
def inverse_interpolation_root(y_vals, x_vals):
    coef = newton_divided_diff(x_vals, y_vals)
    def f(y): return evaluate_newton_polynomial(coef, y_vals, y) - y
    sol = root_scalar(f, bracket=[0.55, 0.58], method='bisect')
    return sol.root if sol.converged else None

x_approx = inverse_interpolation_root(y_vals, x_vals)
true_root = 0.56714329

print(f"三點反插值結果: x ≈ {x_approx:.6f}")
print(f"真實解: x ≈ {true_root}")
print(f"誤差: {abs(x_approx - true_root):.6e}")
