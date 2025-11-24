import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

WIDTH = 1800
HEIGHT = 1600
MAX_ITER = 5000

X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.5, 1.5


# -------------------------------------------------------------------
# Mandelbrot core (Numba-accelerated, parallel)
# -------------------------------------------------------------------

@njit(fastmath=True, parallel=True)
def compute_mandelbrot(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Compute Mandelbrot escape iteration counts for a given window
    of the complex plane.

    This version uses Numba with parallel loops (prange) so it can
    take advantage of multiple CPU cores.
    """
    image = np.zeros((height, width), dtype=np.int32)

    dx = (x_max - x_min) / (width - 1)
    dy = (y_max - y_min) / (height - 1)

    for i in prange(height):
        y = y_min + i * dy
        for j in range(width):
            x = x_min + j * dx

            cr = x
            ci = y

            zr = 0.0
            zi = 0.0
            it = 0

            # zr^2 + zi^2 <= 4  <=>  |z| <= 2
            while (zr * zr + zi * zi) <= 4.0 and it < max_iter:
                zr2 = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr2
                it += 1

            image[i, j] = it

    return image


# -------------------------------------------------------------------
# Coloring and plotting
# -------------------------------------------------------------------

def colorize(image, max_iter):
    """
    Apply a simple log-based coloring to the raw iteration counts.
    Points that never escaped (== max_iter) are treated as "inside".
    """
    img = image.astype(float).copy()

    # Points inside the set (never escaped)
    inside_mask = (img == max_iter)
    img[inside_mask] = 0.0

    # Log scale for smoother gradients on the outside
    nonzero_mask = (img > 0)
    img[nonzero_mask] = np.log(img[nonzero_mask])

    return img


def plot_mandelbrot(img, x_min, x_max, y_min, y_max):
    """
    Display the Mandelbrot image using matplotlib with a nice colormap.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(
        img,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="magma",
    )
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")
    plt.title("Colored Mandelbrot Set (Numba accelerated)")
    plt.colorbar(label="Iterations (log-scaled)")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main():
    # Compute the Mandelbrot set
    image = compute_mandelbrot(
        WIDTH,
        HEIGHT,
        MAX_ITER,
        X_MIN,
        X_MAX,
        Y_MIN,
        Y_MAX,
    )

    # Map iteration counts to something we can feed to a colormap
    img = colorize(image, MAX_ITER)

    # Render the result
    plot_mandelbrot(img, X_MIN, X_MAX, Y_MIN, Y_MAX)


if __name__ == "__main__":
    main()
