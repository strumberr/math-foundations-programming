
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import random
from math import *
import multiprocessing


class Minima:
    def __init__(self, fig, axs):
        self.fig = fig
        self.axs = axs

    def function(self, x):
        return x**4 - 2*x**2 - x + 1
    
    def derivative(self, x):
        return 4*x**3 - 4*x - 1
    
    def function_xy(self, x, y):
        return (x-2)**2+y**2-2*x+3
    
    def derivative_xy(self, x, y):
        return 2*x-4
    
    def minima(self, x, step_size=0.1, iterations=100):

        for i in range(iterations):
            x = x - step_size * self.derivative(x)
            print(x)
            self.axs[0].scatter(x, self.function(x))

            if i > 0:

                if x > x_old:
                    self.axs[0].plot([x_old, x], [self.function(x_old), self.function(x)], color="green", alpha=0.3)
                else:
                    self.axs[0].plot([x_old, x], [self.function(x_old), self.function(x)], color="red", alpha=0.3)

                if x - x_old < 1e-6:
                    break

            plt.pause(0.01)

            x_old = x

    def minimaXY(self, x, y, step_size=0.1, iterations=100):
    
        for i in range(iterations):

            x = x - step_size * self.derivative_xy(x, y)
            y = y - step_size * self.derivative_xy(x, y)
            print(x, y)
            self.axs[1].scatter(x, self.function(x))
            self.axs[1].scatter(y, self.function(y))

            if i > 0:

                if x > x_old:
                    self.axs[1].plot([x_old, x], [self.function(x_old), self.function(x)], color="green", alpha=0.3)
                else:
                    self.axs[1].plot([x_old, x], [self.function(x_old), self.function(x)], color="red", alpha=0.3)

                if x - x_old < 1e-5:
                    break

            plt.pause(0.01)

            x_old = x
    

# ------------------ CLASS BREAK ------------------


class Maxima():
    def __init__(self, fig, axs):
        self.fig = fig
        self.axs = axs


# ------------------ CLASS BREAK ------------------


class MountMain(Minima, Maxima):
    def __init__(self, fig, axs):
        super().__init__(fig, axs)

        self.fig = fig
        self.axs = axs
    

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(6, 6))

    mm = MountMain(fig, axs)

    mm.minimaXY(1, 2, step_size=0.1, iterations=100)
    mm.minima(1, step_size=0.01, iterations=100)
    plt.show()
