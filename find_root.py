import math

def find_root_function(f, a, b, max_iterations=1000, eps=1e-6):

    if f(a) * f(b) >= 0:
        print("The function must have different signs at a and b")

    for el in range(max_iterations):

        c = (a + b) / 2

        if abs(f(c)) < eps:
            return c

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        print(f"Current value: {f(c)}")
        print(f"Current interval: [{a}, {b}]")
        print(f"Current interval: {c}")
        print(f"Current iteration: {el} \n")

    return (a + b) / 2

# def example_function(x):
#     return x**2 - 4

# def example_function_2(x):
#     return x**3 - 3*x + 1

def example_function_3(x):

    return x**4 - 5*x**3 + 2*x**2 - 3*x + 1

def example_function_4(x):
    return math.sin(x) - x**2 + 1


root = find_root_function(example_function_4, -10, 10)
print(f"Groot: {root}")
