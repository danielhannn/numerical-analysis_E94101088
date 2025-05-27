import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

# 題目提供的數據
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# (a) 二次多項式最小平方法
p2 = Polynomial.fit(x, y, 2)         # degree 2
y_fit_poly2 = p2(x)
error_poly2 = np.sum((y - y_fit_poly2)**2)

# (b) y = b * e^(a * x)
def exp_model(x, a, b):
    return b * np.exp(a * x)

params_exp, _ = curve_fit(exp_model, x, y, p0=(0.1, 1.0))
y_fit_exp = exp_model(x, *params_exp)
error_exp = np.sum((y - y_fit_exp)**2)

# (c) y = b * x^a
def power_model(x, a, b):
    return b * x**a

params_pow, _ = curve_fit(power_model, x, y, p0=(2.0, 1.0))
y_fit_pow = power_model(x, *params_pow)
error_pow = np.sum((y - y_fit_pow)**2)

print("Degree-2 Polynomial Coefficients:", p2.convert().coef)
print("Error (Poly Degree 2):", error_poly2)

print("Exponential Fit Parameters (a, b):", params_exp)
print("Error (Exponential Fit):", error_exp)

print("Power Fit Parameters (a, b):", params_pow)
print("Error (Power Fit):", error_pow)



