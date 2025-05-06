import numpy as np

# 定義矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

# 計算反矩陣
A_inv = np.linalg.inv(A)

# 顯示反矩陣
print("Inverse of matrix A is:")
print(np.round(A_inv, 4))  # 四捨五入顯示小數點後四位
