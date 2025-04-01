import numpy as np
from scipy.interpolate import CubicHermiteSpline

# 給定的數據
T = np.array([0, 3, 5, 8, 13])       # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 位置 (英尺)
V = np.array([75, 77, 80, 74, 72])     # 速度 (英尺/秒)

# 建立 Hermite 插值多項式 (位置插值，導數即為速度)
spline = CubicHermiteSpline(T, D, V)

# (a) 預測 t = 10 秒時的車輛位置與速度
t_target = 10
position_at_10 = spline(t_target)
speed_at_10 = spline.derivative()(t_target)

print("== (a) t = 10 秒時的預測 ==")
print(f"位置：{position_at_10:.2f} 英尺")
print(f"速度：{speed_at_10:.2f} 英尺/秒")
print()

# (b) 判斷是否超過 55 英哩/小時（轉換為英尺/秒）
speed_limit = 55 * 5280 / 3600  # 55 mph 換算成 ft/s
# 在觀察區間內以較密取樣點估計速度
t_dense = np.linspace(T[0], T[-1], 1000)
speed_dense = spline.derivative()(t_dense)

above_limit = np.where(speed_dense > speed_limit)[0]
if above_limit.size > 0:
    t_exceed = t_dense[above_limit[0]]
    print("== (b) 超速情形判定 ==")
    print(f"55 英哩/小時約為 {speed_limit:.2f} 英尺/秒")
    print(f"第一次超過該速度限制發生在 t = {t_exceed:.2f} 秒")
else:
    print("== (b) 超速情形判定 ==")
    print("車輛速度未曾超過 55 英哩/小時的限制")
print()

# (c) 預測車輛的最大速度及其發生時間
max_speed = np.max(speed_dense)
t_max_speed = t_dense[np.argmax(speed_dense)]

print("== (c) 車輛最大速度預測 ==")
print(f"預測最大速度為 {max_speed:.2f} 英尺/秒，發生在 t = {t_max_speed:.2f} 秒")
