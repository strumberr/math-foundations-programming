
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

rand_gaussian_array = [[random.randint(1, 100) for i in range(101)] for j in range(100)]
print(f"random matrix: {rand_gaussian_array}")

def generate_hilbert_matrix(n=50):
    
    hilbert_matrix = []
    for i in range(n):
        row = []
        for j in range(n+1):
            element = 1 / (i + j + 1)
            row.append(element)
        hilbert_matrix.append(row)

    return hilbert_matrix

def generate_complex_matrix(n=50):
    
    matrix = np.random.rand(n, n+1)

    for i in range(n):
        for j in range(n+1):
            if np.random.rand() < 0.1:
                matrix[i, j] *= 1e10 if np.random.rand() > 0.5 else 1e-10
    
    return matrix

def generate_alternating_matrix(n=50):
    matrix = np.zeros((n, n+1))

    for i in range(n):
        for j in range(n+1):
            if (i + j) % 2 == 0:
                matrix[i, j] = np.random.rand() * 1e10
            else:
                matrix[i, j] = np.random.rand() * 1e-10

    return matrix

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


# -------------- here we are calculating the distance between the real solution and the estimated solution -------------- #
def calculate_distance(real_solution, estimated_solution):
    difference_vector = np.array(real_solution) - np.array(estimated_solution)
    distance = np.linalg.norm(difference_vector)
    return distance
# -------------- here we are calculating the distance between the real solution and the estimated solution -------------- #

def gaussian_elimination(coefficient_array, constant_array):
    
    print(f"split CoeffArr:")
    for i in range(len(coefficient_array)):
        print(coefficient_array[i])
    print(f"split ConstArr: {constant_array}")
    print("\n")
    
    print(f"i1: {i}")

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

    print(f"i2: {i}")


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

    print(f"i3: {i}")
    
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
n_col_rows = 50
gaussian_array = generate_alternating_matrix(n_col_rows)

for i in range(len(gaussian_array)):
    coefficient_array.append(gaussian_array[i][:-1])
    constant_array.append(gaussian_array[i][-1])


constant_array_copy = constant_array.copy()
coefficient_array_copy = []
for i in range(len(coefficient_array)):
    coefficient_array_copy.append(coefficient_array[i].copy())

x_value, coefficient_array, constant_array = gaussian_elimination(coefficient_array, constant_array)
lhs, rhs = verify_solution(coefficient_array, constant_array, x_value)
solution_difference_rhs_lhs = abs(lhs - rhs)

numpy_solution = np.linalg.solve(coefficient_array_copy, constant_array_copy)
distance_to_real_solution = calculate_distance(numpy_solution, x_value)


print(f"\n\033[93mDistance To Real Solution: \033[0m{distance_to_real_solution}")
print(f"\033[93mLeftHandSide: \033[0m{lhs}")
print(f"\033[93mRightHandSide: \033[0m{rhs}")

print(f"\033[93mStorm solution: \033[0m {x_value}")
print(f"\033[93mNumpy solution: \033[0m{numpy_solution}\n")

if abs(lhs - rhs) > 1e-9:
    print("\n\033[91mThe solution is incorrect.\033[0m")
    print(f"Solution difference: \033[91m{solution_difference_rhs_lhs:.24f}\033[0m")

else:
    print("\n\033[92mThe solution is correct.\033[0m")
    print(f"Solution difference: \033[91m{solution_difference_rhs_lhs:.24f}\033[0m")



num_variables = len(x_value)
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.35
indices = np.arange(num_variables)

rects1 = ax.bar(indices, x_value, bar_width, alpha=0.2, label='custom gaussian elimination', color='blue')
rects2 = ax.bar(indices + bar_width, numpy_solution, bar_width, alpha=0.2, label='numpy method', color='red')

ax.plot(indices, x_value, marker='o', color='blue', label='Custom Gaussian Elimination')
ax.plot(indices, numpy_solution, marker='o', color='red', label='NumPy linalg.solve')

ax.set_xlabel('Variables')
ax.set_ylabel('Solutions')
ax.set_title('comparison of both solutions')
ax.set_xticks(indices + bar_width / 2)
ax.set_xticklabels([f'x{i+1}' for i in range(num_variables)])
ax.grid(True)
ax.legend()

plt.show()