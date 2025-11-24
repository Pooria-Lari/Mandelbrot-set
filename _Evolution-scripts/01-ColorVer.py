import numpy as np
import matplotlib.pyplot as plt

#confige
width  = 800
height = 600
max_iter = 500

x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

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

    #Progress bar
    progress = (i + 1) / height * 100
    print(f"\r Process {progress:.1f}% compelte", end="", flush=True)


# Color
img = image.astype(float).copy()
inside_mask = (img == max_iter)
img[inside_mask] = 0.0

nonzero_mask = (img > 0)
img[nonzero_mask] = np.log(img[nonzero_mask])


#show
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
