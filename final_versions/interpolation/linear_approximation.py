
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import random
from math import *
from sympy import symbols, diff, sympify, solve

class CubicInterpolation:
    def __init__(self):
        self.array_coords = [[0, 1], [1, 4], [2, 2], [3, 0], [4, 1]]
        self.start_time = None

    def numpify_array(self, array_coords=None):

        if array_coords is not None: self.array_coords = array_coords

        x = np.array([coord[0] for coord in self.array_coords])
        y = np.array([coord[1] for coord in self.array_coords])

        x = np.asfarray(x)
        y = np.asfarray(y)

        self.x = x
        self.y = y

        return x, y

    
    def start_and_stop_timer(self):
        
        if self.start_time is None:
            self.start_time = time.time()
            print(f"Timer started at {self.start_time}")
        else:
            stop_time = time.time()
            print(f"Timer stopped at {stop_time}")
            print(f"Elapsed time: {stop_time - self.start_time}")
            total_time = stop_time - self.start_time
            self.start_time = None
            return total_time

    def linear_approximation(self, x=None, y=None):
        c0, c1 = symbols('c0 c1')

        loss_function = 0
        for i in range(len(x)):
            loss_function += (y[i] - (c0 * x[i] + c1))**2

        h = 1e-6

        # derivative_c0 = (loss_function.subs(c0, c0 + h) - loss_function.subs(c0, c0 - h)) / (2 * h)
        # derivative_c1 = (loss_function.subs(c1, c1 + h) - loss_function.subs(c1, c1 - h)) / (2 * h)

        derivative_c0 = diff(loss_function, c0)
        derivative_c1 = diff(loss_function, c1)

        solutions = solve([derivative_c0, derivative_c1], [c0, c1])

        return solutions[c0], solutions[c1]

    

# array_coords = [[0, 10], [1, -5], 
#                 [2, 15], [3, -10], 
#                 [4, 20], [5, 0], 
#                 [6, 25], [7, -20], 
#                 [8, 30], [9, -15], 
#                 [10, 35], [11, -25]]

def cubic_function(x):
    return x**3 - 2*x**2 + x - 1

def sin_function(x):
    return sin(x)

def other_function(x):
    return 1/(1+x**2)

def very_obscure_function(x):
    return 1/(1+25*x**2)

def butterfly_curve(x):
    return np.sin(x) * (np.exp(np.cos(x)) - 2 * np.cos(4 * x) - np.sin(x / 12)**5), np.cos(x) * (np.exp(np.cos(x)) - 2 * np.cos(4 * x) - np.sin(x / 12)**5)

def extra(x):
    return x/2 +1


array_coords = []

for i in np.linspace(-2, 3, 20):

    array_coords.append([i, extra(i)])


for i in range(len(array_coords)):
    array_coords[i][1] += random.randint(-5, 5)



line = CubicInterpolation()
x, y = line.numpify_array(array_coords)

c0, c1 = line.linear_approximation(x, y)


fig, axs = plt.subplots(1, 1, figsize=(14, 6))

plt.scatter(line.x, line.y, color='red', label='Data points')
plt.plot(line.x, [c0 * xi + c1 for xi in line.x], label=f'Best fit line y={c0}*x+{c1}')
plt.grid(True)
plt.legend()
plt.show()

