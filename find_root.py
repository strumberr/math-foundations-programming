
def find_root_function(f, a, b, max_iterations=1000):

    if f(a) * f(b) >= 0:
        print("The function must have different signs at a and b")

    for el in range(max_iterations):

        c = (a + b) / 2

        if f(c) == 0:
            return c

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        print(f"Current interval: {c}")
        print(f"current iteration: {1000 - el}")

    return (a + b) / 2



def example_function(x):
    return x**2 - 4

def example_function_2(x):
    return x**3 - 3*x + 1


def example_function_3(x):

    return x**4 - 5*x**3 + 2*x**2 - 3*x + 1


root = find_root_function(example_function_3, -4, 4)
print(f"Groot: {root}")
