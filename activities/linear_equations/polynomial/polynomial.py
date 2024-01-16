
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import random


class Interpolation:
    def __init__(self, initial_set_xy=[[1, 2], [2, 3], [3, 4]]):

        self.set_xy = np.array(initial_set_xy)
        self.x_array = self.set_xy[:, 0]
        self.y_array = self.set_xy[:, 1]
        self.n = len(self.x_array)

    def solveSystem(self):
        coefficients = []
        
        for i in range(self.n):
            current_y = self.y_array[i]
            coefficient = 1

            for j in range(self.n):
                if i != j:
                    coefficient *= (self.x_array[i] - self.x_array[j]) / (self.x_array[i] - self.x_array[j])

            coefficients.append(current_y * coefficient)

        print(f"Coefficients: {coefficients}")

        return coefficients


interpolation = Interpolation()
coeffs = interpolation.solveSystem()



# Plotting the interpolation
x = np.linspace(min(interpolation.x_array), max(interpolation.x_array), 100)
y = np.polyval(coeffs, x)

plt.plot(x, y)
plt.scatter(interpolation.x_array, interpolation.y_array, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation')
plt.grid(True)
plt.show()
