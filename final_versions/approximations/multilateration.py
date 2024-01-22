
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
from numpy.linalg import lstsq



def read_csv(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

data = read_csv('data_multilateration2.csv')
x_known = data[:, 0]
y_known = data[:, 1]
distances = data[:, 2]

n = len(x_known)

A = np.zeros((n-1, 2))
for i in range(1, n):
    A[i-1, 0] = 2 * (x_known[i] - x_known[0])
    A[i-1, 1] = 2 * (y_known[i] - y_known[0])

print(f"A: {A}")

b = np.zeros((n-1, 1))
for i in range(1, n):
    b[i-1, 0] = distances[0]**2 - distances[i]**2 - x_known[0]**2 - y_known[0]**2 + x_known[i]**2 + y_known[i]**2

print(f"b: {b}")

solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
x_u = solution[0, 0]
y_u = solution[1, 0]

print(f"Solution: {solution}")

plt.scatter(x_known, y_known, label='Known Points')
plt.scatter(x_u, y_u, color='red', label='Unknown Point')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Multilateration Result Data 2')
plt.legend()

for i in range(len(x_known)):
    plt.plot([x_known[i], x_u], [y_known[i], y_u], 'k--', alpha=0.3, linewidth=0.5, label='Distance', color='blue')

plt.show()