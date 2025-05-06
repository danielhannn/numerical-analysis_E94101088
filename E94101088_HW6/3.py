import numpy as np

# 三對角矩陣 A 的下對角、對角、上對角
a = np.array([-1, -1, -1], dtype=float)  # 下對角元素（從第2行開始）
b = np.array([3, 3, 3, 3], dtype=float)  # 對角元素
c = np.array([-1, -1, -1], dtype=float)  # 上對角元素（到倒數第2行止）
d = np.array([2, 3, 4, 1], dtype=float)  # 常數項向量

n = len(b)
l = np.zeros(n)
u = np.zeros(n-1)
y = np.zeros(n)
x = np.zeros(n)

# Crout 分解
l[0] = b[0]
u[0] = c[0] / l[0]

for i in range(1, n-1):
    l[i] = b[i] - a[i-1] * u[i-1]
    u[i] = c[i] / l[i]

l[n-1] = b[n-1] - a[n-2] * u[n-2]

# 前向替代求 y（Ly = d）
y[0] = d[0] / l[0]
for i in range(1, n):
    y[i] = (d[i] - a[i-1] * y[i-1]) / l[i]

# 後向替代求 x（Ux = y）
x[-1] = y[-1]
for i in reversed(range(n-1)):
    x[i] = y[i] - u[i] * x[i+1]

# 顯示解
for i, xi in enumerate(x, start=1):
    print(f"x{i} = {xi:.6f}")
