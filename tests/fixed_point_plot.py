import matplotlib.pyplot as plt
import numpy as np


def find_fixed_point(f, x0, alpha, max_iterations=1000, eps=1e-6):
    for el in range(max_iterations):
        x1 = x0 + alpha * (f(x0) - x0)
        if abs(x1 - x0) < eps:
            return x1, el
        x0 = x1
    return x1, max_iterations

# The function for which we are finding the fixed point
def example_function_7(x):
    return np.sqrt(x + 2)

# Find the fixed point for the function
fixed_point, iterations = find_fixed_point(example_function_7, 2.5, 1)

# Define the range for x values
x_values = np.linspace(0, 5, 400)
y_values = example_function_7(x_values)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=r"$y = \sqrt{x + 2}$")
plt.plot(x_values, x_values, label="y = x", linestyle='--')
plt.scatter([fixed_point], [fixed_point], color='red')
plt.text(fixed_point, fixed_point, f'  Fixed Point ({fixed_point:.2f}, {fixed_point:.2f})', verticalalignment='bottom')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Fixed Point Visualization (Found in {iterations} iterations)')
plt.legend()
plt.grid(True)
plt.show()
