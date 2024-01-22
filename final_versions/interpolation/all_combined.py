
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import random
from math import *
from sympy import symbols, diff, sympify, solve


class AllCombined:
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


    def cubic_interpolate(self, x0=None, x=None, y=None):

        if x0 is None: x0 = self.x0
        if x is None: x = self.x
        if y is None: y = self.y

        xdiff = np.diff(x)
        dydx = np.diff(y) / xdiff

        n = size = len(x)

        w = np.empty(n-1)
        z = np.empty(n)

        w[0] = 0.
        z[0] = 0.

        for i in range(1, n-1):
            m = xdiff[i-1] * (2 - w[i-1]) + 2 * xdiff[i]
            w[i] = xdiff[i] / m
            z[i] = (6 * (dydx[i] - dydx[i-1]) - xdiff[i-1] * z[i-1]) / m

        z[-1] = 0.

        for i in range(n-2, -1, -1):
            z[i] -= w[i] * z[i+1]

        index = np.clip(x.searchsorted(x0), 1, size-1) 

        xi1, xi0 = x[index], x[index-1]
        yi1, yi0 = y[index], y[index-1]
        zi1, zi0 = z[index], z[index-1]
        hi1 = xi1 - xi0

        f0 = (zi0 * (xi1 - x0) ** 3) / (6 * hi1) + \
            (zi1 * (x0 - xi0) ** 3) / (6 * hi1) + \
            ((yi1 / hi1) - (zi1 * hi1 / 6)) * (x0 - xi0) + \
            ((yi0 / hi1) - (zi0 * hi1 / 6)) * (xi1 - x0)
        

        return f0
    

    def lagrange_interpolation(self, x, x_knots, y_knots):
        n = len(x_knots)
        result = 0

        for i in range(n):
            current_y = y_knots[i]
            current_x = x_knots[i]

            for j in range(n):
                if i != j:
                    current_y *= (x - x_knots[j]) / (current_x - x_knots[j])

            result += current_y

        return result
    
    
    def p2p_linear_interpolation(self, x0=None, x=None, y=None):

        if x0 is None: x0 = self.x0
        if x is None: x = self.x
        if y is None: y = self.y

        size = len(x)

        index = np.clip(x.searchsorted(x0), 1, size-1) 

        xi1, xi0 = x[index], x[index-1]
        yi1, yi0 = y[index], y[index-1]
        hi1 = xi1 - xi0

        f0 = yi0 + (x0 - xi0) * (yi1 - yi0) / hi1

        return f0
    

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

def random_function(x):
    return random.randint(-10, 20)


array_coords = []

# for i in range(20):
#     array_coords.append([i, random.randint(-10, 20)])

for i in np.linspace(-2, 3, 500):  # 20 points between -2 and 3

    array_coords.append([i, cubic_function(i)])


cube = AllCombined()
x, y = cube.numpify_array(array_coords)


fig, axs = plt.subplots(1, 3, figsize=(14, 6))
# plt.scatter(x, y)


axs[0].scatter(x, y, label='Data points', color='black', alpha=1)



x_new_cubic = np.linspace(min(x), max(x), 1000)
cube.start_and_stop_timer()
y_new = cube.cubic_interpolate(x_new_cubic, x, y)
total_time1 = cube.start_and_stop_timer()
axs[0].plot(x_new_cubic, y_new, label='cubic_interpolate ', alpha=0.6, color='green')



x_new_lagrange = np.linspace(min(x), max(x), 1000)
cube.start_and_stop_timer()
y_new = cube.lagrange_interpolation(x_new_lagrange, x, y)
total_time2 = cube.start_and_stop_timer()
axs[0].plot(x_new_lagrange, y_new, label='lagrange_interpolation ', alpha=0.6, color='orange')



x_new_linear = np.linspace(min(x), max(x), 1000)
cube.start_and_stop_timer()
y_new = cube.p2p_linear_interpolation(x_new_linear, x, y)
total_time3 = cube.start_and_stop_timer()
axs[0].plot(x_new_linear, y_new, label='p2p_linear_interpolation ', alpha=0.6, color='blue')



x_new_scipy = np.linspace(min(x), max(x), 1000)
cube.start_and_stop_timer()
f = CubicSpline(x, y, bc_type='natural')
total_time4 = cube.start_and_stop_timer()
axs[0].plot(x_new_scipy, f(x_new_scipy), label='CubicSpline SciPy', alpha=0.6, color='red')



x_new_cubic = np.linspace(min(x), max(x), 1000)
cube.start_and_stop_timer()
c0, c1 = cube.linear_approximation(x, y)
total_time5 = cube.start_and_stop_timer()
axs[0].plot(cube.x, [c0 * xi + c1 for xi in cube.x], label='linear_interpolation ', alpha=0.6, color='blue')



axs[1].bar(['cubic_intpl', 'lagrange_intpl', 'p2p_linear_intpl', 'cubic_SciPy_intpl', 'linear_intpl'], [total_time1, total_time2, total_time3, total_time4, total_time5])
axs[1].set_title('Time taken for each method')
axs[1].set_xlabel('Method')
axs[1].set_ylabel('Time taken')
axs[1].set_ylim(0, 0.01)


axs[1].legend()
axs[0].legend()
axs[0].set_ylim(-20, 20)



# Show the plot
plt.show()




plt.show()