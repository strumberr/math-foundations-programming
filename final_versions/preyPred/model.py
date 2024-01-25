
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import random
from math import *
import csv
import networkx as nx
import matplotlib.animation as animation

# def predatorPreyModelWithTime(x, y, alpha, beta):
#     # x is prey, y is predator
#     dxdt = 0.5 * x - 0.05 * x * y
#     dydt = -0.75 * y + 0.025 * x * y

#     return dxdt, dydt

def predatorPreyModel(x, y, alpha, beta, gamma, delta):
    # x is prey, y is predator
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return dxdt, dydt


def eulerMethodWithTime(x0, y0, alpha, beta, gamma, delta, h, time_days):
    x = x0
    y = y0
    total_time = 1

    x_list = [x0]
    y_list = [y0]

    for i in range(time_days):
        dxdt, dydt = predatorPreyModel(x, y, alpha, beta, gamma, delta)
        x += h * dxdt
        y += h * dydt

        x_list.append(x)
        y_list.append(y)

        total_time += 1

    return x_list, y_list, total_time


gamma = 0.03
delta = 0.01
alpha = 0.1
beta = 0.02

x0 = gamma / delta
y0 = alpha / beta

x0 += 0.6
y0 += 0.6

# x0 *= 0.1
# y0 *= 0.1

h = 0.02
time_days = 10000

prey = x0
predator = y0

num_prey, num_predator, time = eulerMethodWithTime(prey, predator, alpha, beta, gamma, delta, h, time_days)

print("Prey:", num_prey)
print("\n")
print("Predator:", num_predator)

plt.plot(range(time), num_prey, label="Prey", color="green")
plt.plot(range(time), num_predator, label="Predator", color="red")
plt.xlabel("Time")
plt.ylabel("Number of animals")
plt.title("Predator-Prey Model")
plt.grid(True)
plt.show()
