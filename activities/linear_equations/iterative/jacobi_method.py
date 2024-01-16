
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify
import random
import sys
import time

from class_this_one_gaussian_good import GaussianElimination


class IterativeSolution:
    def __init__(self):
        initial_matrix = [[5, 1, 1, 7], [1, 4, -1, 4], [1, 2, 5, 8]]
        self.matrix = np.array(initial_matrix)

    def diagonal_matrix(self, n=3):

        diagonal_matrix = []

        for i in range(n-1):
            temp_matrix = []

            for j in range(n):
                if j == i:
                    temp_matrix.append(random.randint(10, 20))
                else:
                    temp_matrix.append(random.randint(0, 10))

            diagonal_matrix.append(temp_matrix)

        diagonal_matrix_copy = diagonal_matrix.copy()


        return diagonal_matrix
    
    def round_each_element(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = round(matrix[i][j], 2)
        
        return matrix
    
    def verify_solution(self, coefficient_array, constant_array, root):
        
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
        

    def split_matrix(self, matrix):
                
        coefficient_array = []
        constant_array = []

        for i in range(len(matrix)):
            coefficient_array.append(matrix[i][:-1])
           #constant_array.append(gaussian_array[i][-1])

        for i in range(len(matrix)):
            sum_all = sum(matrix[i][:-1])
            constant_array.append(sum_all)


        return coefficient_array, constant_array
    
    
    def calculate_residual(self, estimated_solution, coefficient_array_copy, constant_array_copy):
        a = np.array(coefficient_array_copy)
        b = np.array(constant_array_copy)
        residual = np.dot(a, estimated_solution) - b
        residual_error = np.linalg.norm(residual)

        return residual, residual_error
    
    
    def jacobi_method(self, a, b, max_iterations=3000, eps=1e-9):
        
        current_value_array = []
        n = len(a)
        x = [0 for i in range(n)]
        x_new = x.copy()
        k = 0
        stop = False

        while not stop and k < max_iterations:

            for i_el_coeff_array in range(n):

                sigma = 0

                for j_el_coeff_array in range(n):

                    if j_el_coeff_array != i_el_coeff_array:
                        
                        sigma += a[i_el_coeff_array][j_el_coeff_array] * x[j_el_coeff_array]

                    
                x_new[i_el_coeff_array] = (b[i_el_coeff_array] - sigma) / coefficient_array[i_el_coeff_array][i_el_coeff_array]
            
            if np.linalg.norm(np.array(x_new) - np.array(x)) < eps:
                stop = True

            k += 1
            x = x_new.copy()
            current_value_array.append(x_new.copy())

            

        return x_new, k, current_value_array
        



iterative_solution = IterativeSolution()
diagonal_matrix = iterative_solution.diagonal_matrix(n=4)
coefficient_array, constant_array = iterative_solution.split_matrix(diagonal_matrix)

start_time = time.time()
root, iterations, current_value_array = iterative_solution.jacobi_method(coefficient_array, constant_array)
residual, residual_error = iterative_solution.calculate_residual(root, coefficient_array, constant_array)

# print(f"Residual: {residual}")
# print(f"Residual error: {residual_error}")
# print(f"Diagonal MAtrix: {diagonal_matrix}")

# print(f"\nApproximations history: {current_value_array}")

print(f"\nRoot found: {root} in {iterations} iterations")

if iterative_solution.verify_solution(coefficient_array, constant_array, root):
    print("Solution verified")

#stop timer
end_time = time.time()
total_time_jacobi = end_time - start_time


# initalizing gaussian elimination class
gaussian_elimination = GaussianElimination()
print(f"\nMatrix: {diagonal_matrix}")
coefficient_array, constant_array = gaussian_elimination.init_split(diagonal_matrix)
print(f"\Coefficiant array1: {coefficient_array}")
print(f"\nConstant array1: {constant_array}")

start_time = time.time()
coefficient_array_copy, constant_array_copy = gaussian_elimination.copy_arrays(coefficient_array, constant_array)

x_value, coefficient_array, constant_array = gaussian_elimination.main_gaussian(coefficient_array, constant_array)
lhs, rhs = gaussian_elimination.verify_solution(coefficient_array, constant_array, x_value)
solution_difference_rhs_lhs = abs(lhs - rhs)

end_time = time.time()
total_time_gaussian = end_time - start_time


fig, axs = plt.subplots(2, 1, figsize=(6, 8))

axs[0].plot(range(iterations), current_value_array, label='Jacobi Method')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Approximations')
axs[0].legend()

axs[1].bar(['Jacobi Method', 'Gaussian Elimination'], [total_time_jacobi, total_time_gaussian])
axs[1].set_xlabel('Methods')
axs[1].set_ylabel('Time (seconds)')

plt.show()