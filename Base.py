import numpy as np
import matplotlib.pyplot as plt

width = 800
height = 600

max_iter = 300

x_min = -2.0
x_max = 1.0

y_min = -1.5
y_max = 1.5

image = np.zeros((height, width))

for i in range(height):
    y = y_min + i * (y_max - y_min) / (height - 1)
    for j in range(width):
        x = x_min + j * (x_max - x_min) / (width - 1)
        c = complex(x, y)

        z = 0 + 0j
        iter_count = 0
        while abs(z) <= 2 and iter_count < max_iter:
            z = z*z + c
            iter_count += 1
        image[i, j] = iter_count


plt.figure(figsize=(8, 6))
plt.imshow(
    image,
    extent=(x_min, x_max, y_min, y_max),
    origin="lower",
    cmap="magma"
)
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.title("Mandelbrot Set (escape-time coloring)")
plt.colorbar(label="Number of iterations")
plt.show()