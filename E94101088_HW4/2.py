import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

# 原始函數
def f(x):
    return x**2 * np.log(x)

# 區間
a, b = 1, 1.5

# 將函數映射到 [-1, 1]
def mapped_f(t):
    x = (b - a) / 2 * t + (b + a) / 2
    return f(x) * (b - a) / 2

# Gaussian Quadrature 實作
def gaussian_quadrature(n):
    nodes, weights = leggauss(n)
    return np.sum(weights * mapped_f(nodes))

# 精確值
exact_value, _ = quad(f, a, b)

# 計算 n = 3 和 n = 4 的近似值
approx_n3 = gaussian_quadrature(3)
approx_n4 = gaussian_quadrature(4)

# 輸出結果
print(f"Gaussian Quadrature (n=3): {approx_n3:.7f}")
print(f"Gaussian Quadrature (n=4): {approx_n4:.7f}")
print(f"Exact Value:               {exact_value:.7f}")
print(f"Error (n=3):               {abs(approx_n3 - exact_value):.6e}")
print(f"Error (n=4):               {abs(approx_n4 - exact_value):.6e}")
