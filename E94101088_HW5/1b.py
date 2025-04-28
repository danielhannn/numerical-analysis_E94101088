import numpy as np
import matplotlib.pyplot as plt

# 定義 f(t, y)
def f(t, y):
    return 1 + (y/t) + (y/t)**2

# 定義 df/dt 和 df/dy
def df_dt(t, y):
    return -(y / t**2) - 2*(y**2) / t**3

def df_dy(t, y):
    return (1/t) + (2*y) / (t**2)

# 真實解
def exact_solution(t):
    return t * np.tan(np.log(t))

# Taylor's method of order 2
def taylor2_method(f, df_dt, df_dy, t0, y0, h, t_end):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        f_value = f(t, y)
        ft_value = df_dt(t, y) + df_dy(t, y) * f_value
        y_values[i] = y + h*f_value + (h**2/2)*ft_value

    return t_values, y_values

# 初始條件
t0 = 1
y0 = 0
h = 0.1
t_end = 2

# 計算 Taylor2 方法的近似值
t_taylor, y_taylor = taylor2_method(f, df_dt, df_dy, t0, y0, h, t_end)

# 計算真實解
y_exact = exact_solution(t_taylor)

# 顯示比較
import pandas as pd
df = pd.DataFrame({
    "t": t_taylor,
    "Taylor Order 2 Approx": y_taylor,
    "Exact Solution": y_exact,
    "Absolute Error": np.abs(y_exact - y_taylor)
})
print(df)

# 繪圖
plt.figure(figsize=(10,6))
plt.plot(t_taylor, y_taylor, 'o-', label="Taylor Order 2 Approximation")
plt.plot(t_taylor, y_exact, 's--', label="Exact Solution")
plt.xlabel('t')
plt.ylabel('y')
plt.title("Taylor Method Order 2 vs Exact Solution")
plt.legend()
plt.grid(True)
plt.show()
