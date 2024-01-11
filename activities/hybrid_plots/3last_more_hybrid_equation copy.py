
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



def fixed_point_iteration_iter(f, x0, alpha, eps=1e-9):
    print("using fixed point")

    x1 = x0 + alpha * f(x0)
    
    return x1


def da_coolest_function(f, df, x0, alpha=1, max_iterations=1000, eps=1e-6):

    newtons_array = []
    fixed_point_array = []
    x_newton = x0
    x_fixed = x0
    previous_x_newton = x0
    previous_x_fixed = x0


    for _ in range(max_iterations):
        x_newton = newtons_method_iter(f, df, x_newton, alpha, eps)
        newtons_array.append(x_newton)
        
        change = abs(x_newton - previous_x_newton)
        if change > eps and change < 1e-4:
            break

        previous_x_newton = x_newton


    for _ in range(max_iterations):
        x_fixed = fixed_point_iteration_iter(f, x_fixed, alpha, eps)
        fixed_point_array.append(x_fixed)
        
        change = abs(x_fixed - previous_x_fixed)
        if change > eps and change < 1e-4:
            break
        previous_x_fixed = x_fixed

    return x_newton, x_fixed, newtons_array, fixed_point_array

        



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
alpha = 0.3
max_iterations = 1000

x_newton, x_fixed, newtons_array, fixed_point_array = da_coolest_function(example_function, derivative_function, x0, alpha, max_iterations, eps)
print(f"Newton's Method: {newtons_array}")
print(f"Fixed Point Iteration: {fixed_point_array}")
print(f"Fixed point newton: {float(x_newton)}")
print(f"Fixed point: {float(x_fixed)}")



plt.plot(newtons_array, label='Newton\'s Method')
plt.plot(fixed_point_array, label='Fixed Point Iteration')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Comparison of Newton\'s Method and Fixed Point Iteration')
plt.legend()
plt.show()
