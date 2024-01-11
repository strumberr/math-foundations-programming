
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify



def find_fixed_point_newtons_method(f, df, x0, alpha, max_iterations=1000, eps=1e-9):

    print("using newtons method")

    current_value_array = []

    for el in range(max_iterations):
        derivative_function_x0 = df(x0)
        
        if derivative_function_x0 == 0:
            break

        x1 = x0 - alpha * f(x0) / derivative_function_x0
        current_value_array.append(x1)

        if abs(x1 - x0) < eps:
            return x1, el, current_value_array

        x0 = x1

        if el == 999:
            print("reached max iterations")
            break

        # print(f"iteration: {el}")


    return x1, max_iterations, current_value_array


def find_fixed_point(f, x0, alpha, max_iterations=1000, eps=1e-6):
    # sys.set_int_max_str_digits(int(1e6))

    print("using fixed point")

    current_value_array = []

    for el in range(max_iterations):

        x1 = x0 + alpha * (f(x0) - x0)
        current_value_array.append(f(x1))

        if abs(x1 - x0) < eps:
            return x1, el, current_value_array
        
        x0 = x1

        if el == 999:
            print("reached max iterations")
            break

        # print(f"iteration: {el}")



    return x1, max_iterations, current_value_array

def da_coolest_function(example_function, derivative_function, x0, alpha=1, max_iterations=1000, eps=1e-6):


    try:
        root, iterations, current_value_array = find_fixed_point_newtons_method(example_function, 
                                                                                    derivative_function, 
                                                                                    x0, alpha, max_iterations, 
                                                                                    eps)
            
        print(f"\nFixed point: {root}")
        print(f"Iterations: {iterations}")
        print(f"Approximations history: {current_value_array}\n")

        
    except:
        try:
            root, iterations, current_value_array = find_fixed_point(example_function, x0, 
                                                                 alpha, max_iterations, 
                                                                 eps)
        
            print(f"\nFixed point: {root}")
            print(f"Iterations: {iterations}")
            print(f"Approximations history: {current_value_array}\n")

        except:
            print(f"couldn't find root")

    return root, iterations, current_value_array


# def example_function_7(x):
#     return np.sqrt(x + 2)*2

x = symbols('x')

def example_function(x):

    equation = (x-3)*(x-1)**2
    return equation

def derivative_function(x_value):
    
    derivative_expr = diff(example_function(x), x)
    derivative_value = derivative_expr.subs(x, x_value)

    return derivative_value


x0 = float(2)
eps = 1e-9
alpha = 1
max_iterations = 1000

root, iterations, current_value_array = da_coolest_function(example_function, derivative_function, x0, alpha, max_iterations, eps)

error_array = np.abs(np.array(current_value_array) - root)

x_values = np.linspace(-20, 10, 400)
y_values = example_function(x_values)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

# root = root / 2

axs[0].plot(x_values, y_values, label="Function: cos(x) - 1")
axs[0].plot(x_values, x_values, label="y = x", linestyle='--')
axs[0].scatter([root], [example_function(root)], color='red')
axs[0].text(root, example_function(root), f'  Root ({root:.4f}, {example_function(root):.4f})', verticalalignment='bottom')
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

axs[2].plot(error_array, label="Error", color='green')
axs[2].set_xlabel("Iterations")
axs[2].set_ylabel("Error")
axs[2].set_title("Error Over Iterations")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

