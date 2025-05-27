import numpy as np
from scipy.integrate import quad

# Function to approximate
def f(x):
    return x**2 * np.sin(x)

# Number of sample points
m = 16
x_samples = np.linspace(0, 1, m, endpoint=False)
f_samples = f(x_samples)

# Compute coefficients for S_4
a0 = (1 / m) * np.sum(f_samples)

a_k = []
b_k = []
for k in range(1, 5):  # k = 1 to 4
    cos_term = np.cos(2 * np.pi * k * x_samples)
    sin_term = np.sin(2 * np.pi * k * x_samples)
    a_k.append((2 / m) * np.sum(f_samples * cos_term))
    b_k.append((2 / m) * np.sum(f_samples * sin_term))

# Define S_4(x)
def S4(x):
    result = a0
    for k in range(1, 5):
        result += a_k[k-1] * np.cos(2 * np.pi * k * x)
        result += b_k[k-1] * np.sin(2 * np.pi * k * x)
    return result

# Part (b): Compute ∫₀¹ S₄(x) dx
integral_S4, _ = quad(S4, 0, 1)

# Part (c): Compute ∫₀¹ x² sin(x) dx (actual)
integral_actual, _ = quad(f, 0, 1)

# Part (d): Compute error E(S₄) = ∫₀¹ (f(x) - S₄(x))² dx
error_integrand = lambda x: (f(x) - S4(x))**2
error, _ = quad(error_integrand, 0, 1)

# Output results
print("Integral of S4(x) over [0,1]:", integral_S4)
print("Actual integral of x^2 * sin(x) over [0,1]:", integral_actual)
print("Error E(S4):", error)
