
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import random


class Interpolation:
    def __init__(self):
        pass

    def poly(self, coeffs, x):
        
        result = 0
        t = 1

        for i in range(len(coeffs)):
            result += coeffs[i] * t
            t *= x

        return result
    

    def build_interpolating_polynomial(self, x_knots, y_knots):
        n = len(x_knots)
        a = np.zeros((n, n))
        b = np.array(y_knots)

        for i in range(n):
            a[i] = np.power(x_knots[i], range(n))

        coeffs = np.linalg.solve(a, b)

        return coeffs

        
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

        
        


# coeffs = [1, 0, 1]
    
x_knots = [0, 1, 2, -2, 5]
y_knots = [1, 2, 5, 1, 4]

input_x = 1

interpolation = Interpolation()

coeffs = interpolation.build_interpolating_polynomial(x_knots, y_knots)
print(coeffs)
# coeffs = [1, 0, 1, 2, 5]

x_values = np.linspace(min(x_knots), max(x_knots), 100)

y1 = [interpolation.poly(coeffs, x) for x in x_values]
print(y1)
y2 = interpolation.lagrange_interpolation(x_values, x_knots, y_knots)

plt.plot(x_values, y1, 'r', label='Interpolation polynomial', linewidth=2, marker='o', markerfacecolor='blue', markersize=3)
plt.plot(x_values, y2, 'b', label='Lagrange interpolation', linewidth=2)

plt.scatter(x_knots, y_knots, color='black', label='Data points')

plt.grid()
plt.legend()
plt.show()
