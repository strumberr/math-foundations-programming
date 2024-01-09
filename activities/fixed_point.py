
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
    return np.sqrt(x + 2)



fixed_point, iterations, current_value_array = find_fixed_point(example_function_7, 400, 0.2)
print(f"Fixed point: {fixed_point}")
print(f"Iterations: {iterations}")
print(f"Value history: {current_value_array}")

x_values = np.linspace(-3, 5, 400)
y_values = example_function_7(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="plot")
plt.plot(x_values, x_values, label="y = x", linestyle='--')
plt.scatter([fixed_point], [fixed_point], color='red')
plt.text(fixed_point, fixed_point, f'  Fixed Point ({fixed_point:.2f}, {fixed_point:.2f})', verticalalignment='bottom')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Fixed Point Visualization (Found in {iterations} iterations)')
plt.legend()
plt.grid(True)
plt.show()