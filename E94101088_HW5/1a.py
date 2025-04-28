import numpy as np
import matplotlib.pyplot as plt

# 定義微分方程 y' = 1 + (y/t) + (y/t)^2
def f(t, y):
    return 1 + (y/t) + (y/t)**2

# 真實解 y(t) = t * tan(ln(t))
def exact_solution(t):
    return t * np.tan(np.log(t))

# Euler's method
def euler_method(f, t0, y0, h, t_end):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])

    return t_values, y_values

# 初始條件
t0 = 1
y0 = 0
h = 0.1
t_end = 2

# 計算 Euler 方法的近似值
t_approx, y_approx = euler_method(f, t0, y0, h, t_end)

# 計算真實解
y_exact = exact_solution(t_approx)

# 顯示比較
import pandas as pd
df = pd.DataFrame({
    "t": t_approx,
    "Euler Approximation": y_approx,
    "Exact Solution": y_exact,
    "Absolute Error": np.abs(y_exact - y_approx)
})
print(df)

# 繪圖
plt.figure(figsize=(10,6))
plt.plot(t_approx, y_approx, 'o-', label="Euler's Method Approximation")
plt.plot(t_approx, y_exact, 's--', label="Exact Solution")
plt.xlabel('t')
plt.ylabel('y')
plt.title("Euler's Method vs Exact Solution")
plt.legend()
plt.grid(True)
plt.show()
