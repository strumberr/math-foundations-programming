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

x = np.linspace(0, 2*np.pi, 100)
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





# ---------------------------- butterfly curve ----------------------------
# def butterfly_curve(t):
#     x = np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
#     y = np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
#     return x, y

# t_samples_reduced = np.linspace(0, 12 * np.pi, 10)
# xy_samples_reduced = np.array([butterfly_curve(t) for t in t_samples_reduced])

# interpolation_x_reduced = Interpolation(np.column_stack((t_samples_reduced, xy_samples_reduced[:, 0])))
# interpolation_y_reduced = Interpolation(np.column_stack((t_samples_reduced, xy_samples_reduced[:, 1])))

# t_interp_fine = np.linspace(0, 12 * np.pi, 1000)
# x_interp_fine = np.array([interpolation_x_reduced.lagrange_interpolation(ti) for ti in t_interp_fine])
# y_interp_fine = np.array([interpolation_y_reduced.lagrange_interpolation(ti) for ti in t_interp_fine])

# plt.figure(figsize=(10, 6))
# plt.plot(x_interp_fine, y_interp_fine, label='Interpolated Butterfly Curve', color='blue')
# plt.scatter(xy_samples_reduced[:, 0], xy_samples_reduced[:, 1], color='red', label='Sample Points')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Interpolation of Butterfly Curve')
# plt.legend()
# plt.grid(True)
# plt.show()
# ---------------------------- butterfly curve ----------------------------