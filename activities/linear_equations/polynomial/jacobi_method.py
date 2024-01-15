
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import random



class IterativeSolution:
    def __init__(self):
        initial_matrix = [[5, 1, 1, 7], [1, 4, -1, 4], [1, 2, 5, 8]]
        self.matrix = np.array(initial_matrix)
