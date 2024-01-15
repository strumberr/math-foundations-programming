
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import random


# I commented the code so its more readable, i hope it helps!
class GaussianElimination():
    def __init__(self):

        self.gaussian_array = [[1, 1, 1, 3], 
                        [2, -1, -1, 0], 
                        [6, -4, 2, 4]]


    def generate_hilbert_matrix(self, n=50):
        
        hilbert_matrix = []
        for i in range(n):
            row = []
            for j in range(n+1):
                element = 1 / (i + j + 1)
                row.append(element)
            hilbert_matrix.append(row)

        return hilbert_matrix

    def generate_complex_matrix(self, n=50):
        
        matrix = np.random.rand(n, n+1)

        for i in range(n):
            for j in range(n+1):
                if np.random.rand() < 0.1:
                    matrix[i, j] *= 1e10 if np.random.rand() > 0.5 else 1e-10
        
        return matrix

    def generate_alternating_matrix(self, n=50):
        matrix = np.zeros((n, n+1))

        for i in range(n):
            for j in range(n+1):
                if (i + j) % 2 == 0:
                    matrix[i, j] = np.random.rand() * 1e10
                else:
                    matrix[i, j] = np.random.rand() * 1e-10

        return matrix
    
    def generate_random_matrix(self, n=50):
        rand_gaussian_array = [[random.randint(1, 100) for i in range(n+1)] for j in range(n)]
        return rand_gaussian_array
    
    def choose_matrix(self, matrix_type, n=20):
        print(f"Matrix type: {n}")
        if matrix_type == "hilbert":
            return self.generate_hilbert_matrix(n)
        elif matrix_type == "complex":
            return self.generate_complex_matrix(n)
        elif matrix_type == "alternating":
            return self.generate_alternating_matrix(n)
        elif matrix_type == "random":
            return self.generate_random_matrix(n)
        else:
            return self.gaussian_array
    
    def init_split(self, gaussian_array):
                
        coefficient_array = []
        constant_array = []

        for i in range(len(gaussian_array)):
            coefficient_array.append(gaussian_array[i][:-1])
           #constant_array.append(gaussian_array[i][-1])

        for i in range(len(gaussian_array)):
            sum_all = sum(gaussian_array[i][:-1])
            constant_array.append(sum_all)


        return coefficient_array, constant_array
    
    def copy_arrays(self, coefficient_array, constant_array):

        constant_array_copy = constant_array.copy()
        coefficient_array_copy = []
        
        for i in range(len(coefficient_array)):
            coefficient_array_copy.append(coefficient_array[i].copy())
        
        return coefficient_array_copy, constant_array_copy
    

    def verify_solution_sum(self, coefficient_array, constant_array, root):
        
        for i in range(len(coefficient_array)):
            sigma = 0

            for j in range(len(coefficient_array[i])):
                sigma += coefficient_array[i][j] * root[j]
            
            # print(f"Sigma: {sigma} - Constant: {constant_array[i]}")

            difference = abs(sigma - constant_array[i])

            # print(f"Difference: {difference}")
            
            if difference > 1e-6:
                return False

        return True
            

    # -------------- here we are verifiing the solution -------------- #
    def verify_solution(self, coefficient_array, constant_array, solution_vector):
        
        for i in range(len(coefficient_array)):
            lhs = 0
            for j in range(len(coefficient_array[i])):
                lhs += coefficient_array[i][j] * solution_vector[j]
            rhs = constant_array[i]
        return lhs, rhs
    # -------------- here we are verifiing the solution -------------- #


    # -------------- here we are calculating the distance between the real solution and the estimated solution -------------- #
    def calculate_distance(self, real_solution, estimated_solution):
        difference_vector = np.array(real_solution) - np.array(estimated_solution)
        distance = np.linalg.norm(difference_vector)
        return distance
    # -------------- here we are calculating the distance between the real solution and the estimated solution -------------- #

    def calculate_residuals_and_absolute_errors(self, real_solution, estimated_solution, coefficient_array_copy, constant_array_copy):
        a = np.array(coefficient_array_copy)
        b = np.array(constant_array_copy)
        residual = np.dot(a, estimated_solution) - b
        residual_error = np.linalg.norm(residual)

        return residual, residual_error


    # -------------- #here we are calculating the factor by which the current row will be multiplied and then subtracted from the row below it -------------- #
    def first_step(self, coefficient_array, constant_array, i):
        coefficient_array = coefficient_array
        constant_array = constant_array

        for column in range(0, min(len(coefficient_array), len(coefficient_array[0]))):

            for row_below_current_one in range(column+1, len(gaussian_array)):

                factor = coefficient_array[row_below_current_one][column] / coefficient_array[column][column]

                # print(f"coefficient_array[i][iteration]: {coefficient_array[row_below_current_one][column]}")
                # print(f"coefficient_array[iteration][iteration]: {coefficient_array[column][column]}")
                # print(f"factor: {factor}")
                
                for el_current_row in range(len(coefficient_array[i])):

                    coefficient_array[row_below_current_one][el_current_row] -= factor * coefficient_array[column][el_current_row]

                constant_array[row_below_current_one] -= factor * constant_array[column]

        return coefficient_array, constant_array, i
    
    # -------------- #here we are calculating the factor by which the current row will be multiplied and then subtracted from the row below it -------------- #

    # -------------- pretty matrix print -------------- #
    def pretty_matrix_print(self, coefficient_array, constant_array):

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
    # -------------- pretty matrix print -------------- #
            

            


    def solutionMatrix(self, coefficient_array, constant_array, i):
        # -------------- x will hold the solutions to the system of equations -------------- #
        x = [0 for i in range(len(coefficient_array))]
        
        for i in range(len(coefficient_array)):
            print()

        print(f"i: {i}")

        for rows_reverse in range(len(coefficient_array)-1, -1, -1):

            if (coefficient_array[rows_reverse][rows_reverse] == 0) or (constant_array[rows_reverse] == 0):
                x[rows_reverse] = max(constant_array[rows_reverse], coefficient_array[rows_reverse][rows_reverse])
            else:
                x[rows_reverse] = constant_array[rows_reverse] / coefficient_array[rows_reverse][rows_reverse]
            
            for k in range(i-1, -1, -1):
                constant_array[k] -= coefficient_array[k][rows_reverse] * x[rows_reverse]
        
        return x, coefficient_array, constant_array, i
        # -------------- x will hold the solutions to the system of equations -------------- #

    

    def main_gaussian(self, coefficient_array, constant_array):
        
        i = 0

        coefficient_array, constant_array, i = self.first_step(coefficient_array, constant_array, i)

        self.pretty_matrix_print(coefficient_array, constant_array)
        
        x, coefficient_array, constant_array, i = self.solutionMatrix(coefficient_array, constant_array, i)

        return x, coefficient_array, constant_array





n_iterations = 50

array_matrixes = ["hilbert", "complex", "alternating", "random"]

results = {matrix: {"residual_error": [], "absolute_error": [], "dimensionality": []} for matrix in array_matrixes}


fig, axs = plt.subplots(1, 3, figsize=(14, 6))

for current_matrix in array_matrixes:

    residual_error_final_array = []
    absolute_error_final_array = []
    dimensionality_array = []


    for el_num, n_col_rows in enumerate(range(1, n_iterations)):

        n_col_rows = n_col_rows

        gaussian_elimination = GaussianElimination()

        # you can choose between "hilbert", "complex", "alternating", "random" or "gaussian_array"
        gaussian_array = gaussian_elimination.choose_matrix(current_matrix, n_col_rows)

        coefficient_array, constant_array = gaussian_elimination.init_split(gaussian_array)

        
        coefficient_array_copy, constant_array_copy = gaussian_elimination.copy_arrays(coefficient_array, constant_array)


        x_value, coefficient_array, constant_array = gaussian_elimination.main_gaussian(coefficient_array, constant_array)
        lhs, rhs = gaussian_elimination.verify_solution(coefficient_array, constant_array, x_value)
        solution_difference_rhs_lhs = abs(lhs - rhs)


        numpy_solution = np.linalg.solve(coefficient_array_copy, constant_array_copy)

        # distance_to_real_solution = gaussian_elimination.calculate_distance(numpy_solution, x_value)
        residuals, residual_errors = gaussian_elimination.calculate_residuals_and_absolute_errors(numpy_solution, x_value, coefficient_array_copy, constant_array_copy)

        distance_to_real_solution = gaussian_elimination.calculate_distance(numpy_solution, x_value)

        results[current_matrix]["residual_error"].append(residual_errors)
        results[current_matrix]["dimensionality"].append(n_col_rows)

        results[current_matrix]["absolute_error"].append(distance_to_real_solution)
        

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


# Plotting the results for each matrix type
for matrix in array_matrixes:
    axs[0].plot(results[matrix]["dimensionality"], results[matrix]["residual_error"], label=f"Residual Error {matrix}")
    axs[1].plot(results[matrix]["dimensionality"], results[matrix]["absolute_error"], label=f"Absolute Error {matrix}")

# Setting labels and title
axs[0].set_xlabel("Dimensionality")
axs[0].set_ylabel("Residual Error")
axs[0].set_title("Residual Error by Matrix Type")
axs[0].legend()

axs[1].set_xlabel("Dimensionality")
axs[1].set_ylabel("Absolute Error")
axs[1].set_title("Absolute Error by Matrix Type")
axs[1].legend()

axs[2].set_xlabel("Dimensionality")
axs[2].set_ylabel("Distance To Real Solution")
axs[2].set_title("Distance To Real Solution by Matrix Type")
axs[2].legend()




plt.show()
