
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np


def find_fixed_point(f, x0, alpha, max_iterations=1000, eps=1e-6):
    # sys.set_int_max_str_digits(int(1e6))

    current_value_array = []

    for el in range(max_iterations):

        x1 = x0 + alpha * (f(x0) - x0)
        current_value_array.append(f(x1))

        if abs(x1 - x0) < eps:
            return x1, el, current_value_array
        
        x0 = x1

        print(f"Current value: {f(x1)}")
        print(f"Current iteration: {el + 1} \n")



    return x1, max_iterations, current_value_array

def example_function(x):
    return x**2 - 4

def example_function_2(x):
    return x**3 - 3*x + 1

def example_function_3(x):
    return sqrt(1)

def example_function_4(x):

    return x**4 - 5*x**3 + 2*x**2 - 3*x + 1

def example_function_5(x):
    return math.sin(x) - x**2 + 1

def example_function_6(x):
    return x**2 - 2

def example_function_7(x):
    return np.sqrt(x + 2)*2

def example_function_8(x):
    return x**2 - 4



fixed_point, iterations, current_value_array = find_fixed_point(example_function_7, 10, 1)
print(f"Fixed point: {fixed_point}")
print(f"Iterations: {iterations}")
print(f"Value history: {current_value_array}")

x_values = np.linspace(-3, 10, 400)
y_values = example_function_7(x_values)


fig, axs = plt.subplots(2, 1, figsize=(6, 8))


axs[0].plot(x_values, y_values, label="plot")
axs[0].plot(x_values, x_values, label="y = x", linestyle='--')
axs[0].scatter([fixed_point], [fixed_point], color='red')
axs[0].text(fixed_point, fixed_point, f'  Fixed Point ({fixed_point:.2f}, {fixed_point:.2f})', verticalalignment='bottom')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title(f'Fixed Point Visualization (Found in {iterations} iterations)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(iterations + 1), current_value_array, marker='o')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Value')
axs[1].set_title('Iteration History')
axs[1].grid(True)

plt.tight_layout()
plt.show()