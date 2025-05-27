import numpy as np

# Define the function
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

# Generate sample points in the interval [-1, 1]
x = np.linspace(-1, 1, 100)
y = f(x)

# Fit a degree 2 least squares polynomial
coeffs = np.polyfit(x, y, 2)

# Extract coefficients a0, a1, a2
a2, a1, a0 = coeffs  # 注意順序：polyfit 回傳的是最高次項在前

# Print the coefficients
print(f"a0 = {a0}")
print(f"a1 = {a1}")
print(f"a2 = {a2}")
