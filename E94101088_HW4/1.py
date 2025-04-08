import numpy as np

# Composite Simpson's Rule
def composite_simpson(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = f(x[0]) + f(x[-1])
    for i in range(1, n):
        coef = 4 if i % 2 == 1 else 2
        result += coef * f(x[i])
    return (h / 3) * result


def f_a(x):
    return np.where(x == 0, 0, x**(-1/4) * np.sin(x))

epsilon = 1e-8
a_result = composite_simpson(f_a, epsilon, 1, 4)


def f_b_fixed(t):
    return np.where(t == 0, 0, t**2 * np.sin(1 / t))

b_result = composite_simpson(f_b_fixed, 1e-8, 1, 4)

# 輸出結果
print("a) ≈ {:.7f}".format(a_result))
print("b) ≈ {:.7f}".format(b_result))