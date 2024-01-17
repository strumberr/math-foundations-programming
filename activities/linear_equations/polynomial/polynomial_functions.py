import numpy as np
import matplotlib.pyplot as plt

class Interpolation:
    def __init__(self, initial_set_xy):
        self.set_xy = np.array(initial_set_xy)
        self.x_array = self.set_xy[:, 0]
        self.y_array = self.set_xy[:, 1]
        self.n = len(self.x_array)

    def lagrange_interpolation(self, x):

        result = 0

        for i in range(self.n):

            term = self.y_array[i]

            for j in range(self.n):

                if j != i:

                    term = term * (x - self.x_array[j]) / (self.x_array[i] - self.x_array[j])
            
            result += term
            
        return result


x_samples = np.linspace(0, 2*np.pi, 10)
y_samples = np.sin(x_samples)

interpolation = Interpolation(np.column_stack((x_samples, y_samples)))

x = np.linspace(0, 2*np.pi, 10)
y_interpolated = np.array([interpolation.lagrange_interpolation(xi) for xi in x])
y_actual = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_interpolated, label='Interpolated', color='blue')
plt.plot(x, y_actual, label='Actual Sine Function', color='green', linestyle='--')
plt.scatter(x_samples, y_samples, color='red', label='Sample Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation of Sine Function (but using few points)')
plt.legend()
plt.grid(True)
plt.show()
