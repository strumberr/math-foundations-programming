
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify



def newtons_method_iter(f, df, x0, alpha, eps=1e-9):
    print("using newtons method")
    derivative = df(x0)

    if abs(derivative) < eps:
        return x0

    x1 = x0 - alpha * f(x0) / derivative

    return x1


def fixed_point_iteration_iter(f, x0, alpha, eps=1e-6):
    print("using fixed point")
    x1 = x0 + alpha * (f(x0) - x0)

    return x1


def choose_method(f, df, x0, previous_value, eps):
    derivative = df(x0)
    change = abs(x0 - previous_value)


    if abs(derivative) < eps:
        return "fixed_point"


    if change > eps and change < 1e-4:
        return "fixed_point" if change > 0.5 * abs(previous_value - x0) else "newton"


    return "newton"

def da_coolest_function(f, df, x0, alpha=1, max_iterations=1000, eps=1e-6):
    
    previous_value = x0
    root_array = []
    root_dict = {}

    for el in range(max_iterations):

        method = choose_method(f, df, x0, previous_value, eps)

        if method == "newton":
            x0 = newtons_method_iter(f, df, x0, alpha, eps)
        else:
            x0 = fixed_point_iteration_iter(f, x0, alpha, eps)

        x0 = round(x0, 20)

        root_dict[f"{method}_{el}"] = x0
        
        root_array.append(x0)

        print(f"iteration: {el}, x0: {x0}")

        if abs(previous_value - x0) < eps:
            break

        if len(root_array) > 1 and abs(root_array[-1] - root_array[-2]) < eps:
            break

        previous_value = x0


        # if abs(derivative) > eps:
        #     x0 = newtons_method_iter(f, df, x0, alpha, eps)
        #     x0 = round(x0, 20)
        #     root_dict[f"newton_{el}"] = x0

        # else:
        #     x0 = fixed_point_iteration_iter(f, x0, alpha, eps)
        #     x0 = round(x0, 20)
        #     root_dict[f"fixed_point_{el}"] = x0


        # if el % 2 == 0:
        #     try:
        #         x0 = newtons_method_iter(f, df, x0, alpha, eps)
        #         x0 = round(x0, 20)
        #         root_dict[f"newton_{el}"] = x0
        #     except:
        #         print("newtons method failed")
        #         break

        # else:
        #     try:
        #         x0 = fixed_point_iteration_iter(f, x0, alpha, eps)
        #         x0 = round(x0, 20)
        #         root_dict[f"fixed_point_{el}"] = x0
        #     except:
        #         print("fixed point iteration failed")
        #         break

      

    return x0, root_array, len(root_array), root_dict



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

root, root_array, n_iterations, root_dict = da_coolest_function(example_function, derivative_function, x0, alpha, max_iterations, eps)
print(f"Fixed point: {float(root)}")
print(f"Approximations history: {root_array}\n")
print(f"Root dictionary: {root_dict}")

error_array = np.abs(np.array(root_array) - root)


x_values = np.linspace(-20, 10, 400)
y_values = example_function(x_values)

fig, axs = plt.subplots(1, 4, figsize=(14, 4))


axs[0].plot(range(len(root_array)), root_array, label="Approximations", marker='o', linestyle='-', color='b')
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Approximation Value")
axs[0].set_title("Root Approximation History")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(error_array, label="Error", color='green')
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Error")
axs[1].set_title("Error Over Iterations")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(x_values, y_values, label="Function: cos(x) - 1")
axs[2].plot(x_values, x_values, label="y = x", linestyle='--')
axs[2].scatter([root], [example_function(root)], color='red')
axs[2].text(root, example_function(root), f'  Root ({root:.4f}, {example_function(root):.4f})', verticalalignment='bottom')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].set_title(f"Newton's Method Root Visualization (Found in {n_iterations} iterations)")
axs[2].legend()
axs[2].grid(True)



iteration_numbers = list(range(len(root_dict)))
root_estimations = list(root_dict.values())
colors = ['red' if 'newton' in key else 'blue' for key in root_dict.keys()]

axs[3].scatter(iteration_numbers, root_estimations, c=colors)
axs[3].set_xlabel('Iteration')
axs[3].set_ylabel('Root Estimate')
axs[3].set_title('Root Estimations by Method')
plt.show()
