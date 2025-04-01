import numpy as np

def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    P_x = 0.0
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if i != j:
                L_i *= (x - x_values[j]) / (x_values[i] - x_values[j])
        P_x += y_values[i] * L_i
    return P_x

# 給定資料點
x_values = np.array([0.698, 0.733, 0.768, 0.803])
y_values = np.array([0.7661, 0.7432, 0.7193, 0.6946])

# 欲近似之目標點
x_target = 0.750

# 插值結果（不使用 chop）
approx_value_1 = lagrange_interpolation(x_values[:2], y_values[:2], x_target)  # 一次
approx_value_2 = lagrange_interpolation(x_values[:3], y_values[:3], x_target)  # 二次
approx_value_3 = lagrange_interpolation(x_values, y_values, x_target)          # 三次

# 真實值
actual_value = 0.7317  # cos(0.750) 給定值

# 實際誤差
actual_error_1 = abs(actual_value - approx_value_1)
actual_error_2 = abs(actual_value - approx_value_2)
actual_error_3 = abs(actual_value - approx_value_3)

# 誤差界計算（使用四階導數的最大值，即 cos(x) 本身）
def fourth_derivative_cos(x):
    return np.cos(x)

fourth_derivative = abs(fourth_derivative_cos(x_target))

# 理論誤差界
error_bound_1 = (fourth_derivative / 24) * np.prod([abs(x_target - xi) for xi in x_values[:2]])
error_bound_2 = (fourth_derivative / 24) * np.prod([abs(x_target - xi) for xi in x_values[:3]])
error_bound_3 = (fourth_derivative / 24) * np.prod([abs(x_target - xi) for xi in x_values])

# 結果輸出
print(f"Lagrange Approximation (1st-degree): {approx_value_1}")
#print(f"Actual Error (1st-degree): {actual_error_1}")
print(f"Error Bound (1st-degree): {error_bound_1}")
print(f"Lagrange Approximation (2nd-degree): {approx_value_2}")
#print(f"Actual Error (2nd-degree): {actual_error_2}")
print(f"Error Bound (2nd-degree): {error_bound_2}")
print(f"Lagrange Approximation (3rd-degree): {approx_value_3}")
#print(f"Actual Error (3rd-degree): {actual_error_3}")
print(f"Error Bound (3rd-degree): {error_bound_3}")
