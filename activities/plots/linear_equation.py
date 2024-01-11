
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import time

x = symbols("x")


# def equation_1():
#     x = symbols("x")
#     f = x ** 3 - 2 * x - 5
#     df = diff(f, x)
#     return f, df

# def equation_2():
#     x = symbols("x")
#     f = x ** 3 - 2 * x - 5
#     df = diff(f, x)
#     return f, df

y1 = x ** 2 + 1
y2 = 2*x + 1

x = np.linspace(-5, 5, 1000)

plt.plot(x, y1, label="y1")
plt.plot(x, y2, label="y2")
plt.grid()
plt.legend()
plt.show()


