import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

WIDTH = 800
HEIGHT = 600
MAX_ITER = 500

X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.5, 1.5


# -------------------------------------------------------------------
# Core: compute a single Mandelbrot row (Numba-accelerated)
# -------------------------------------------------------------------

@njit(fastmath=True)
def compute_row(y, width, max_iter, x_min, x_max):
    """
    Compute one horizontal row of Mandelbrot iteration counts
    for a fixed y value.
    """
    row = np.zeros(width, dtype=np.int32)
    dx = (x_max - x_min) / (width - 1)

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

        row[j] = it

    return row


# -------------------------------------------------------------------
# High-level Mandelbrot computation with a simple progress bar
# -------------------------------------------------------------------

def compute_mandelbrot_with_progress(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Compute the full Mandelbrot image row by row,
    printing a simple text progress indicator.
    """
    image = np.zeros((height, width), dtype=np.int32)
    dy = (y_max - y_min) / (height - 1)

    for i in range(height):
        y = y_min + i * dy

        # Fast row computation via Numba-compiled function
        image[i, :] = compute_row(y, width, max_iter, x_min, x_max)

        # Text-based progress bar
        progress = (i + 1) / height * 100.0
        print(f"\rComputing Mandelbrot... {progress:.1f}% done", end="", flush=True)

    print()  # newline after finishing
    return image


# -------------------------------------------------------------------
# Coloring and plotting
# -------------------------------------------------------------------

def colorize(image, max_iter):
    """
    Map raw iteration counts to a float image suitable for colormaps.
    Uses log scaling for smoother gradients outside the set.
    """
    img = image.astype(float).copy()

    # Points that did not escape are considered "inside" the set
    inside_mask = (img == max_iter)
    img[inside_mask] = 0.0

    # Log scale for all points that escaped
    nonzero_mask = (img > 0)
    img[nonzero_mask] = np.log(img[nonzero_mask])

    return img


def plot_mandelbrot(img, x_min, x_max, y_min, y_max):
    """
    Display the Mandelbrot image using matplotlib.
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
    plt.title("Colored Mandelbrot Set (Numba + progress)")
    plt.colorbar(label="Iterations (log-scaled)")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main():
    image = compute_mandelbrot_with_progress(
        WIDTH,
        HEIGHT,
        MAX_ITER,
        X_MIN,
        X_MAX,
        Y_MIN,
        Y_MAX,
    )

    img = colorize(image, MAX_ITER)
    plot_mandelbrot(img, X_MIN, X_MAX, Y_MIN, Y_MAX)


if __name__ == "__main__":
    main()
