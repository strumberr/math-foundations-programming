
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import time



def newtons_method_iter(f, df, x0, alpha, eps=1e-9):
    print("using newtons method")
    derivative = df(x0)

    x1 = x0 - alpha * f(x0) / derivative

    return x1



# def fixed_point_iteration_iter(f, x0, alpha, eps=1e-9):
#     print("using fixed point")

#     x1 = x0 + alpha * f(x0)
    
#     return x1


# def fixed_point_iteration_iter(f, x0, alpha, eps=1e-9):
#     a = 0
#     b = 1
    
#     if f(a) * f(b) > 0:
#         raise ValueError("Function does not have opposite signs at boundaries")
    
#     x1 = (a + b) / 2.0
    
#     return x1

def fixed_point_iteration_iter(f, a, b, alpha, eps=1e-9):
    if f(a) * f(b) > 0:
        raise ValueError("Function does not have opposite signs at boundaries")
    
    root_array = []

    while (b - a) / 2.0 > eps:
        x1 = (a + b) / 2.0
        
        if f(x1) == 0:
            return x1
        elif f(a) * f(x1) < 0:
            b = x1

        else:
            a = x1

        root_array.append(x1)

        

    return (a + b) / 2.0, root_array




def da_coolest_function(f, df, x0, alpha=1, max_iterations=1000, eps=1e-6):

    newtons_array = []
    fixed_point_array = []
    x_newton = x0
    x_fixed = x0
    previous_x_newton = x0
    previous_x_fixed = x0


    start_time_newton = time.time()

    for _ in range(max_iterations):
        x_newton = newtons_method_iter(f, df, x_newton, alpha, eps)
        newtons_array.append(x_newton)
        
        change = abs(x_newton - previous_x_newton)
        if change > eps and change < 1e-4:
            break

        previous_x_newton = x_newton
    
    end_time_newton = time.time()
    total_time_newton = end_time_newton - start_time_newton


    start_time_fixed = time.time()

    # for _ in range(max_iterations):
    #     x_fixed = fixed_point_iteration_iter(f, x_fixed, alpha, eps)
    #     fixed_point_array.append(x_fixed)
        
    #     change = abs(x_fixed - previous_x_fixed)
    #     if change > eps and change < 1e-4:
    #         break
    #     previous_x_fixed = x_fixed
    a = 0
    b = 1

    x_fixed, bisection_root_array = fixed_point_iteration_iter(f, a, b, alpha, eps)
    
    end_time_fixed = time.time()
    total_time_fixed = end_time_fixed - start_time_fixed

    return x_newton, x_fixed, newtons_array, bisection_root_array, total_time_newton, total_time_fixed

        

# ----------------------------------- custom one -----------------------------------

def custom_newtons_method_iter(f, df, x0, alpha, eps=1e-9):
    print("using newtons method")
    derivative = df(x0)

    if abs(derivative) < eps:
        return x0

    x1 = x0 - alpha * f(x0) / derivative

    return x1


def custom_fixed_point_iteration_iter(f, x0, alpha, eps=1e-6):
    print("using fixed point")
    x1 = x0 + alpha * (f(x0) - x0)

    return x1


def custom_choose_method(f, df, x0, previous_value, eps):
    derivative = df(x0)
    change = abs(x0 - previous_value)


    if abs(derivative) < eps:
        return "fixed_point"


    if change > eps and change < 1e-4:
        return "fixed_point" if change > 0.5 * abs(previous_value - x0) else "newton"


    return "newton"

def custom_main_function(f, df, x0, alpha=1, max_iterations=1000, eps=1e-6):
    
    previous_value = x0
    root_array = []
    root_dict = {}

    start_time_custom = time.time()

    for el in range(max_iterations):

        method = custom_choose_method(f, df, x0, previous_value, eps)

        if method == "newton":
            x0 = custom_newtons_method_iter(f, df, x0, alpha, eps)
        else:
            x0 = custom_fixed_point_iteration_iter(f, x0, alpha, eps)

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
    
    end_time_custom = time.time()
    total_time_custom = end_time_custom - start_time_custom

      

    return x0, root_array, len(root_array), root_dict, total_time_custom


# ----------------------------------- custom one -----------------------------------



# def example_function_7(x):
#     return np.sqrt(x + 2)*2

x = symbols('x')

def example_function(x):

    equation = 1-exp(-x)
    return equation

def derivative_function(x_value):
    
    # derivative_expr = diff(example_function(x), x)
    # derivative_value = derivative_expr.subs(x, x_value)

    return exp(-x_value)


x0 = float(2)
eps = 1e-9
alpha = 0.01
max_iterations = 1000

x_newton, x_fixed, newtons_array, fixed_point_array, total_time_newton, total_time_fixed = da_coolest_function(example_function, derivative_function, x0, alpha, max_iterations, eps)

#----------------------------------- custom one -----------------------------------

x_custom, root_array_custom, iterations_custom, root_dict_custom, total_time_custom = custom_main_function(example_function, derivative_function, x0, alpha, max_iterations, eps)

#----------------------------------- custom one -----------------------------------


print(f"Newton's Method: {newtons_array}")
print(f"Fixed Point Iteration: {fixed_point_array}")
print(f"Custom: {root_array_custom}")

print(f"Fixed point newton: {float(x_newton)}")
print(f"Fixed point: {float(x_fixed)}")
print(f"Fixed point custom: {float(x_custom)}")



# plt.plot(newtons_array, label='Newton\'s Method')
# plt.plot(fixed_point_array, label='Fixed Point Iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Value')
# plt.title('Comparison of Newton\'s Method and Fixed Point Iteration')
# plt.legend()
# plt.show()

error_array_newton = np.abs(np.array(newtons_array) - x_newton)
error_array_fixed = np.abs(np.array(fixed_point_array) - x_fixed)
error_array_custom = np.abs(np.array(root_array_custom) - x_custom)


fig, axs = plt.subplots(1, 3, figsize=(14, 6))


axs[0].plot(newtons_array, label='Newton\'s Method', linestyle='--', linewidth=3, color='red')
axs[0].plot(fixed_point_array, label='Fixed Point Iteration', linestyle='--', linewidth=2, color='green')
axs[0].plot(root_array_custom, label='Custom', linestyle='--', linewidth=1, color='blue')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Value')
axs[0].set_title('Comparison of Methods by Value')
axs[0].legend()
axs[0].grid(True)


axs[1].plot(range(len(newtons_array)), [total_time_newton] * len(newtons_array), label='Newton\'s Method', linestyle='--', linewidth=3, color='red')
axs[1].plot(range(len(fixed_point_array)), [total_time_fixed] * len(fixed_point_array), label='Fixed Point Iteration', linestyle='--', linewidth=2, color='green')
axs[1].plot(range(len(root_array_custom)), [total_time_custom] * len(root_array_custom), label='Custom', linestyle='--', linewidth=1, color='blue')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Time')
axs[1].set_title('Comparison of Methods by Time')
axs[1].grid(True)
axs[1].legend()


axs[2].plot(error_array_newton, label='Newton\'s Method', linestyle='--', linewidth=3, color='red')
axs[2].plot(error_array_fixed, label='Fixed Point Iteration', linestyle='--', linewidth=2, color='green')
axs[2].plot(error_array_custom, label='Custom', linestyle='--', linewidth=1, color='blue')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Error')
axs[2].set_title('Comparison of Methods by Error')
axs[2].legend()
axs[2].grid(True)


plt.tight_layout()
plt.show()