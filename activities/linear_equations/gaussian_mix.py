
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import re 



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


    x = [0 for i in range(len(coefficient_array))]

    for i in range(len(coefficient_array)-1, -1, -1):


        if coefficient_array[i][i] != 0:
            x[i] = constant_array[i] / coefficient_array[i][i]
        else:
            x[i] = max(constant_array[i], coefficient_array[i][i])

        for k in range(i-1, -1, -1):
            constant_array[k] -= coefficient_array[k][i] * x[i]



    print(f"final values: {x}")


def convert_to_gaussian_array_corrected(equations):
    gaussian_array = []

    for eq in equations:
        parts = eq.split()

        row = []

        sign = 1

        for part in parts:
            if part == '+':
                sign = 1
            elif part == '-':
                sign = -1
            elif part == '=':
                sign = 1
            else:
                if 'x' in part:
                    if part[0] == 'x':
                        coef = 1
                    else:
                        coef = int(part.split('x')[0])

                    row.append(coef * sign)
                else:
                    row.append(int(part) * sign)

        gaussian_array.append(row)

    return gaussian_array


equation_array = ["x1 + x2 + x3 = 3", "2x1 - x2 - x3 = 0", "6x1 - 4x2 + 2x3 = 4"]
gaussian_array_result = convert_to_gaussian_array_corrected(equation_array)
gaussian_elimination(gaussian_array_result)



# gaussian_elimination(gaussian_array)