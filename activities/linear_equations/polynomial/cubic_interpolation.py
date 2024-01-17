
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import random

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
    
    
    def linear_interpolation(self, x0=None, x=None, y=None):

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


array_coords = []

for i in range(20):
    array_coords.append([i, random.randint(-10, 20)])


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


cube.start_and_stop_timer()
x_new_lagrange = np.linspace(min(x), max(x), 1000)
y_new = cube.lagrange_interpolation(x_new_lagrange, x, y)
axs[0].plot(x_new_lagrange, y_new, label='lagrange_interpolation ', alpha=0.6, color='orange')
total_time2 = cube.start_and_stop_timer()


cube.start_and_stop_timer()
x_new_linear = np.linspace(min(x), max(x), 1000)
y_new = cube.linear_interpolation(x_new_linear, x, y)
axs[0].plot(x_new_linear, y_new, label='linear_interpolation ', alpha=0.6, color='blue')
total_time3 = cube.start_and_stop_timer()


cube.start_and_stop_timer()
x_new_scipy = np.linspace(min(x), max(x), 1000)
f = CubicSpline(x, y, bc_type='natural')
axs[0].plot(x_new_scipy, f(x_new_scipy), label='CubicSpline SciPy', alpha=0.6, color='red')
total_time4 = cube.start_and_stop_timer()


axs[1].bar(['cubic_intpl', 'lagrange_intpl', 'linear_intpl', 'cubic_SciPy_intpl'], [total_time1, total_time2, total_time3, total_time4])
axs[1].set_title('Time taken for each method')
axs[1].set_xlabel('Method')
axs[1].set_ylabel('Time taken')


axs[1].legend()
axs[0].legend()
axs[0].set_ylim(-30, 40)


plt.show()