import numpy as np

# 系數矩陣 A 和常數向量 b
A = np.array([
    [1.19,   2.11, -100.0,   1.0],
    [14.2,  -0.112, 12.2,   -1.0],
    [0.0,   100.0, -99.9,    0.0],
    [15.3,   0.11, -13.1,   -1.0]
], dtype=float)

b = np.array([1.12, 3.44, 2.15, 4.16], dtype=float)

# 高斯消去法含部分樞軸選擇
n = len(b)

# 前向消去 (Forward elimination)
for i in range(n):
    # 找最大主元進行列交換
    max_row = np.argmax(np.abs(A[i:n, i])) + i
    if i != max_row:
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
    
    # 消去
    for j in range(i+1, n):
        factor = A[j][i] / A[i][i]
        A[j, i:] = A[j, i:] - factor * A[i, i:]
        b[j] = b[j] - factor * b[i]

# 回代 (Back substitution)
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i][i]

# 顯示解
for idx, val in enumerate(x, start=1):
    print(f"x{idx} = {val:.6f}")
