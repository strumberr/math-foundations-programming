
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify


gaussian_array = [[1, 1, 1, 3], [2, -1, -1, 0], [6, -4, 2, 4]]

gaussian_array_2 = [[1, 1, 1, 3, 1], [2, -1, -1, 0, 2], [6, -4, 2, 4, 3], [6, -4, 2, 4, 3]]

def gaussian_elimination(gaussian_array):
    print(f"init GA: {gaussian_array}")
    print("\n")

    coefficient_array = []
    constant_array = []

    for i in range(len(gaussian_array)):
        coefficient_array.append(gaussian_array[i][:-1])
        constant_array.append(gaussian_array[i][-1])

    print(f"split CoeffArr: {coefficient_array}")
    print(f"split ConstArr: {constant_array}")
    print("\n")

    for iteration in range(0, min(len(coefficient_array), len(coefficient_array[0]))):


        for i in range(iteration+1, len(gaussian_array)):

            # print(f"iteration: {iteration}")

            factor = coefficient_array[i][iteration] / coefficient_array[iteration][iteration]

            for j in range(len(coefficient_array[i])):

                coefficient_array[i][j] -= factor * coefficient_array[iteration][j]

            constant_array[i] -= factor * constant_array[iteration]



    print(f"final CoeffArr: {coefficient_array}")
    print(f"final ConstArr: {constant_array}")
    print("\n")

    for i in range(len(coefficient_array)):
        for j in range(len(coefficient_array[i])):
            if str(coefficient_array[i][j]) == "1":
                print(f"x{j + 1}", end=" ")
            else:
                if str(coefficient_array[i][j]) == "0.0":
                    print(f"0", end=" ")
                else:
                    print(f"{int(coefficient_array[i][j])}x{j + 1}", end=" ")


        print(f"= {constant_array[i]}")


gaussian_elimination(gaussian_array_2)