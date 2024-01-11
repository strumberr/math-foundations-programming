
from math import *
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, sympify


equation_array = ["x1 +x2 +x3 = 3", "2x1 -x2 -x3 = 0", "6x1 -4x2 +2x3 = 4"]

coefficients = []
constants = []

for equation in equation_array:
    equation = equation.split("=")

    equation[0] = equation[0].split(" ")
    equation[0].remove("")


    equation[1] = equation[1].replace(" ", "")

    #get the first value of each equation, if the first value is negative or positive, then move to the second value, if its not a number, then it is a 1
    for i in range(len(equation[0])):
        if equation[0][i][0] == "-":
            equation[0][i] = equation[0][i].replace("-", "")
            equation[0][i] = str(equation[0][i]) * -1
        elif equation[0][i][0] == "+":
            equation[0][i] = equation[0][i].replace("+", "")
            equation[0][i] = str(equation[0][i])
        elif equation[0][i][0] == "x":
            equation[0][i] = equation[0][i].replace("x", "")
            equation[0][i] = str(equation[0][i])
        else:
            equation[0][i] = 1




    coefficients.append(equation[0])
    constants.append(equation[1])


print(coefficients)
print(constants)


