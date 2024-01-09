
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np



def find_root_newtons_method(f, df, x0, max_iterations=1000, eps=1e-9):
    current_value_array = []

    for el in range(max_iterations):
        derivative_function_x0 = df(x0)
        
        if derivative_function_x0 == 0:
            break

        x1 = x0 - f(x0) / derivative_function_x0
        current_value_array.append(x1)

        if abs(x1 - x0) < eps:
            return x1, el, current_value_array

        x0 = x1

        print(f"Current approximation: {x1}")
        print(f"Current iteration: {el + 1} \n")

    return x1, max_iterations, current_value_array


def example_function_7(x):
    return x**2 - 4

def derivative_example_function_7(x):
    return 2*x

x0 = 2.9

root, iterations, current_value_array = find_root_newtons_method(example_function_7, derivative_example_function_7, x0)

print(f"Root found: {root}")
print(f"Iterations: {iterations}")
print(f"Approximations history: {current_value_array}")


x_values = np.linspace(-20, 10, 400)
y_values = x_values**2 - 4


fig, axs = plt.subplots(2, 1, figsize=(6, 8))

# root = root / 2

axs[0].plot(x_values, y_values, label="Function: cos(x) - 1")
axs[0].plot(x_values, x_values, label="y = x", linestyle='--')
axs[0].scatter([root], [example_function_7(root)], color='red')
axs[0].text(root, example_function_7(root), f'  Root ({root:.2f}, {example_function_7(root):.2f})', verticalalignment='bottom')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title(f"Newton's Method Root Visualization (Found in {iterations} iterations)")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(current_value_array, label="Approximations", marker='o', linestyle='-', color='b')
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Approximation Value")
axs[1].set_title("Convergence of Newton's Method")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()