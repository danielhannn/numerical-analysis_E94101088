import numpy as np

# 系統的係數矩陣 A 和常數項向量 b
A = np.array([
    [ 4, -1,  0, -1,  0,  0],
    [-1,  4, -1,  0, -1,  0],
    [ 0, -1,  4,  0,  1, -1],
    [-1,  0,  0,  4, -1, -1],
    [ 0, -1,  0, -1,  4, -1],
    [ 0,  0, -1,  0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

def jacobi(A, b, x0, tol=1e-10, max_iter=1000):
    D = np.diag(np.diag(A))
    R = A - D
    x = x0.copy()
    for k in range(max_iter):
        x_new = np.linalg.inv(D).dot(b - R.dot(x))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Jacobi converged in {k+1} iterations.")
            return x_new
        x = x_new
    print("Jacobi did not converge.")
    return x

def gauss_seidel(A, b, x0, tol=1e-10, max_iter=1000):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Gauss-Seidel converged in {k+1} iterations.")
            return x_new
        x = x_new
    print("Gauss-Seidel did not converge.")
    return x

def sor(A, b, x0, omega=1.1, tol=1e-10, max_iter=1000):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = x[i] + omega * ((b[i] - s1 - s2) / A[i, i] - x[i])
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"SOR (ω={omega}) converged in {k+1} iterations.")
            return x_new
        x = x_new
    print("SOR did not converge.")
    return x

def conjugate_gradient(A, b, x0, tol=1e-10, max_iter=1000):
    x = x0.copy()
    r = b - A.dot(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    for k in range(max_iter):
        Ap = A.dot(p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            print(f"Conjugate Gradient converged in {k+1} iterations.")
            return x
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    print("Conjugate Gradient did not converge.")
    return x

# 初始猜測
x0 = np.zeros(6)

# 執行方法並輸出結果
x_jacobi = jacobi(A, b, x0)
print("Jacobi solution:")
print(x_jacobi, "\n")

x_gs = gauss_seidel(A, b, x0)
print("Gauss-Seidel solution:")
print(x_gs, "\n")

x_sor = sor(A, b, x0, omega=1.1)
print("SOR solution:")
print(x_sor, "\n")

x_cg = conjugate_gradient(A, b, x0)
print("Conjugate Gradient solution:")
print(x_cg, "\n")
