
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import random
from math import *

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



array_coords = []

# for i in range(20):
#     array_coords.append([i, random.randint(-10, 20)])

for i in np.linspace(-2, 3, 20):  # 20 points between -2 and 3

    array_coords.append([i, very_obscure_function(i)])


cube = CubicInterpolation()
x, y = cube.numpify_array(array_coords)


fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plt.scatter(x, y)


axs[0].scatter(x, y, label='Data points', color='black', alpha=1)


cube.start_and_stop_timer()
x_new_cubic = np.linspace(min(x), max(x), 100)
y_new = cube.cubic_interpolate(x_new_cubic, x, y)
axs[0].plot(x_new_cubic, y_new, label='cubic_interpolate ', alpha=0.6, color='green')
total_time1 = cube.start_and_stop_timer()



axs[1].bar(['cubic_intpl'], [total_time1])
axs[1].set_title('Time taken for each method')
axs[1].set_xlabel('Method')
axs[1].set_ylabel('Time taken')


axs[1].legend()
axs[0].legend()
# axs[0].set_ylim(-3, 3)


plt.show()