
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import random
import sys
import time
import csv



def f(x):
    return 2 + 3 * sin(x) - x

def phi(x):
    return [1, x, x**2, sin(x), cos(x)]

def g(x, c):
    return c[0] * 1 + c[1] * x + c[2] * x**2 + c[3] * sin(x) + c[4] * cos(x)


def read_csv(filename):

    data = np.loadtxt(filename, delimiter=',')
    
    return data


data = read_csv('data2.csv')
x = data[:, 0]
y = data[:, 1]

phi_array = np.array([phi(x) for x in x])
    
print(f"phi_array: {phi_array}")

coefficients = np.linalg.solve(phi_array.T @ phi_array, phi_array.T @ y)

print(f"coefficients: {coefficients}")

x_plot = np.linspace(min(x), max(x), 100)
y_plot = np.array([g(x, coefficients) for x in x_plot])

plt.plot(x_plot, y_plot)
plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximation')
plt.grid(True)
plt.show()
