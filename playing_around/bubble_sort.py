import time
import matplotlib.pyplot as plt
import random

e = [3, 6, 6, 1, 65, 84, 23, 61, 2, 87, 4, 3, 8, 6, 9, 4, 5, 1, 8]

e = [random.randint(1, 100) for i in range(100)]

def bubble_sort(e):
    fig, ax = plt.subplots()
    ax.set_title("Bubble Sort")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    bar = ax.bar(range(len(e)), e)

    for i in range(len(e)):
        for j in range(len(e)-1):
            if e[j] > e[j+1]:
                e[j], e[j+1] = e[j+1], e[j]

                for rect, val in zip(bar, e):
                    rect.set_height(val)
                plt.pause(0.001)

    return e

print(bubble_sort(e))
plt.show()
