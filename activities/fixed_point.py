#solve fixed point problem
from math import *
import math
import sys

def find_fixed_point(f, x0, max_iterations=1000, eps=1e-6):
    # sys.set_int_max_str_digits(int(1e6))

    for el in range(max_iterations):

        x1 = f(x0)

        if abs(x1 - x0) < eps:
            return x1
        
        x0 = x1

        print(f"Current value: {f(x1)}")
        print(f"Current iteration: {el + 1} \n")

    return x1

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
    return x**2 - 4


fixed_point = find_fixed_point(example_function_5, 0)
print(f"Fixed point: {fixed_point}")