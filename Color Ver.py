import numpy as np
import matplotlib.pyplot as plt


width = 800
height = 600

max_iter = 300

x_min = -2.0
x_max = 1.0

y_min = -1.5
y_max = 1.5


image = np.zeros((height, width), dtype=int)


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





img = image.astype(float).copy()
inside_mask = (img == max_iter)
img[inside_mask] = 0.0

nonzero_mask = (img > 0)
img[nonzero_mask] = np.log(img[nonzero_mask])


plt.figure(figsize=(8, 6))
plt.imshow(
    img,
    extent=(x_min, x_max, y_min, y_max),
    origin="lower",
    cmap="magma"
)
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.title("Colored Mandelbrot Set")
plt.colorbar(label="Iterations (log-scaled)")
plt.show()
