import time
import matplotlib.pyplot as plt
import random

e = [3, 6, 6, 1, 65, 84, 23, 61, 2, 87, 4, 3, 8, 6, 9, 4, 5, 1, 8]

e = [random.randint(1, 100) for i in range(100)]

def binary_sort(e):
    fig, ax = plt.subplots()
    ax.set_title("Binary Sort")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    bar = ax.bar(range(len(e)), e)

    for i in range(1, len(e)):
        key = e[i]
        low = 0
        high = i - 1

        while low <= high:
            mid = (low + high) // 2
            if key < e[mid]:
                high = mid - 1
            else:
                low = mid + 1

        for j in range(i, low, -1):
            e[j] = e[j - 1]

            for rect, val in zip(bar, e):
                rect.set_height(val)
            plt.pause(0.001)

        e[low] = key

        for rect, val in zip(bar, e):
            rect.set_height(val)
        plt.pause(0.001)

    return e


print(binary_sort(e))
plt.show()
