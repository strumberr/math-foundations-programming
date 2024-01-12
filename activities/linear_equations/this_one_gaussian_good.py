
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import random


gaussian_array = [[1, 1, 1, 3], 
                  [2, -1, -1, 0], 
                  [6, -4, 2, 4]]

# gaussian_array_2 = [[1000, 2000, 3000, 4000],
#                   [5000, 6000, 7000, 8000],
#                   [9000, 10000, 11000, 12000]]
# rand_gaussian_array = [[random.randint(1, 100) for i in range(4)] for j in range(3)]
# num_columns = int(input("Enter the number of columns: ")) + 1
# num_rows = int(input("Enter the number of rows: "))
# rand_gaussian_array = [[random.randint(1, 100) for i in range(num_columns)] for j in range(num_rows)]

rand_gaussian_array = [[random.randint(1, 100) for i in range(11)] for j in range(10)]
print(f"random matrix: {rand_gaussian_array}")




# I commented this code so its more readable, i hope it helps!

# -------------- here we are verifiing the solution -------------- #
def verify_solution(coefficient_array, constant_array, solution_vector):
    
    for i in range(len(coefficient_array)):
        lhs = 0
        for j in range(len(coefficient_array[i])):
            lhs += coefficient_array[i][j] * solution_vector[j]
        rhs = constant_array[i]
    return lhs, rhs
# -------------- here we are verifiing the solution -------------- #


def gaussian_elimination(coefficient_array, constant_array):
    
    print(f"split CoeffArr:")
    for i in range(len(coefficient_array)):
        print(coefficient_array[i])
    print(f"split ConstArr: {constant_array}")
    print("\n")

    # -------------- #here we are calculating the factor by which the current row will be multiplied and then subtracted from the row below it -------------- #
    for column in range(0, min(len(coefficient_array), len(coefficient_array[0]))):

        for row_below_current_one in range(column+1, len(gaussian_array)):

            factor = coefficient_array[row_below_current_one][column] / coefficient_array[column][column]

            print(f"coefficient_array[i][iteration]: {coefficient_array[row_below_current_one][column]}")
            print(f"coefficient_array[iteration][iteration]: {coefficient_array[column][column]}")
            print(f"factor: {factor}")
            
            for el_current_row in range(len(coefficient_array[i])):

                coefficient_array[row_below_current_one][el_current_row] -= factor * coefficient_array[column][el_current_row]

            constant_array[row_below_current_one] -= factor * constant_array[column]
    # -------------- #here we are calculating the factor by which the current row will be multiplied and then subtracted from the row below it -------------- #

    print(f"final CoeffArr: {coefficient_array}")
    print(f"final ConstArr: {constant_array}")
    print("\n")

    # -------------- printing the matrix in a nice way -------------- #
    max_widths = [max([len(str(int(round(coefficient_array[i][j])))) for i in range(len(coefficient_array))]) for j in range(len(coefficient_array[0]))]
    max_constant_width = max([len(str(int(constant_array[i]))) for i in range(len(constant_array))])

    for i in range(len(coefficient_array)):

        for j in range(len(coefficient_array[i])):
            coefficient = coefficient_array[i][j]
            rounded_coefficient = int(round(coefficient))
            formatted_coefficient = f"{rounded_coefficient:<{max_widths[j]}}"
            print(formatted_coefficient, end=" ")

        constant = int(constant_array[i])
        formatted_constant = f"= {constant:<{max_constant_width}}"
        print(formatted_constant)
    # -------------- printing the matrix in a nice way -------------- #


    # -------------- x will hold the solutions to the system of equations -------------- #
    x = [0 for i in range(len(coefficient_array))]

    for rows_reverse in range(len(coefficient_array)-1, -1, -1):

        if (coefficient_array[rows_reverse][rows_reverse] == 0) or (constant_array[rows_reverse] == 0):
            x[rows_reverse] = max(constant_array[rows_reverse], coefficient_array[rows_reverse][rows_reverse])
        else:
            x[rows_reverse] = constant_array[rows_reverse] / coefficient_array[rows_reverse][rows_reverse]
        
        for k in range(i-1, -1, -1):
            constant_array[k] -= coefficient_array[k][rows_reverse] * x[rows_reverse]
    
    # -------------- x will hold the solutions to the system of equations -------------- #
    return x, coefficient_array, constant_array


coefficient_array = []
constant_array = []
gaussian_array = rand_gaussian_array

for i in range(len(gaussian_array)):
    coefficient_array.append(gaussian_array[i][:-1])
    constant_array.append(gaussian_array[i][-1])

x_value, coefficient_array, constant_array = gaussian_elimination(coefficient_array, constant_array)
lhs, rhs = verify_solution(coefficient_array, constant_array, x_value)

print(f"\n\033[93mLeftHandSide: \033[0m{lhs}")
print(f"\033[93mRightHandSide: \033[0m{rhs}")

print(f"\033[93mStorm solution: \033[0m {x_value}")
print(f"\033[93mNumpy solution: \033[0m{np.linalg.solve(coefficient_array, constant_array)}\n")

if abs(lhs - rhs) > 1e-9:
    print("\n\033[91mThe solution is incorrect.\033[0m")
else:
    print("\n\033[92mThe solution is correct.\033[0m")

