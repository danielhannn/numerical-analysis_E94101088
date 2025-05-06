import numpy as np

# 定義係數矩陣 A 和常數向量 b
A = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
], dtype=float)

b = np.array([1.12, 3.44, 2.15, 4.16], dtype=float)

# 合併為增廣矩陣
Ab = np.hstack([A, b.reshape(-1,1)])

n = len(b)

# 高斯消去（含部分選主元）
for i in range(n):
    # 部分選主元
    max_row = np.argmax(np.abs(Ab[i:, i])) + i
    Ab[[i, max_row]] = Ab[[max_row, i]]
    
    # 消去過程
    for j in range(i+1, n):
        factor = Ab[j][i] / Ab[i][i]
        Ab[j, i:] -= factor * Ab[i, i:]

# 回代求解
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

print("解為 x =", x)
