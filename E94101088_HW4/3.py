import numpy as np
from scipy.integrate import dblquad
from numpy.polynomial.legendre import leggauss

# 定義被積分函數 f(x, y)
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# ========= 精確值（用 scipy 的 dblquad） =========
exact_value, _ = dblquad(
    func=lambda y, x: f(x, y),
    a=0,
    b=np.pi / 4,
    gfun=lambda x: np.sin(x),
    hfun=lambda x: np.cos(x)
)

# ========= a. Simpson's Rule（n = m = 4） =========
def simpsons_double(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = 0
    for i in range(n + 1):
        xi = x[i]
        wx = 1 if i in [0, n] else (4 if i % 2 == 1 else 2)
        inner_a = np.sin(xi)
        inner_b = np.cos(xi)
        m = n
        k = (inner_b - inner_a) / m
        y = np.linspace(inner_a, inner_b, m + 1)
        inner_result = 0
        for j in range(m + 1):
            yj = y[j]
            wy = 1 if j in [0, m] else (4 if j % 2 == 1 else 2)
            inner_result += wy * f(xi, yj)
        result += wx * (k / 3) * inner_result
    return (h / 3) * result

simpsons_result = simpsons_double(f, 0, np.pi / 4, 4)

# ========= b. Gaussian Quadrature（n = m = 3） =========
def gaussian_double(f, a, b, n):
    nodes, weights = leggauss(n)
    result = 0
    for i in range(n):
        xi = 0.5 * (b - a) * nodes[i] + 0.5 * (b + a)
        wx = weights[i]
        inner_a = np.sin(xi)
        inner_b = np.cos(xi)
        inner_result = 0
        for j in range(n):
            yj = 0.5 * (inner_b - inner_a) * nodes[j] + 0.5 * (inner_b + inner_a)
            wy = weights[j]
            inner_result += wy * f(xi, yj)
        result += wx * (0.5 * (inner_b - inner_a)) * inner_result
    return 0.5 * (b - a) * result

gaussian_result = gaussian_double(f, 0, np.pi / 4, 3)

# ========= 輸出結果 =========
print("Exact Value:                     {:.7f}".format(exact_value))
print("Simpson's Rule (n = m = 4):      {:.7f}".format(simpsons_result))
print("Gaussian Quadrature (n = m = 3): {:.7f}".format(gaussian_result))
print("Error (Simpson):                 {:.2e}".format(abs(simpsons_result - exact_value)))
print("Error (Gaussian):                {:.2e}".format(abs(gaussian_result - exact_value)))
