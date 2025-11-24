import math

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, cuda


# -------------------------------------------------------------------
# Mandelbrot cores: CPU single, CPU multi, GPU
# -------------------------------------------------------------------

@njit(fastmath=True)
def compute_mandelbrot_cpu_single(width, height, max_iter,
                                  x_min, x_max, y_min, y_max):
    """
    Mandelbrot computation on a single CPU core (Numba, no parallel loops).
    """
    image = np.zeros((height, width), dtype=np.int32)

    dx = (x_max - x_min) / (width - 1)
    dy = (y_max - y_min) / (height - 1)

    for i in range(height):
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


@njit(fastmath=True, parallel=True)
def compute_mandelbrot_cpu_multi(width, height, max_iter,
                                 x_min, x_max, y_min, y_max):
    """
    Mandelbrot computation using multiple CPU cores (Numba + prange).
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

            while (zr * zr + zi * zi) <= 4.0 and it < max_iter:
                zr2 = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr2
                it += 1

            image[i, j] = it

    return image


@cuda.jit
def mandelbrot_kernel(x_min, x_max, y_min, y_max, max_iter, image):
    """
    GPU kernel: each CUDA thread computes one pixel.
    """
    i, j = cuda.grid(2)
    height = image.shape[0]
    width = image.shape[1]

    if i < height and j < width:
        dx = (x_max - x_min) / (width - 1)
        dy = (y_max - y_min) / (height - 1)

        x = x_min + j * dx
        y = y_min + i * dy

        cr = x
        ci = y

        zr = 0.0
        zi = 0.0
        it = 0

        while zr * zr + zi * zi <= 4.0 and it < max_iter:
            zr2 = zr * zr - zi * zi + cr
            zi = 2.0 * zr * zi + ci
            zr = zr2
            it += 1

        image[i, j] = it


def compute_mandelbrot_gpu(width, height, max_iter,
                           x_min, x_max, y_min, y_max):
    """
    Mandelbrot computation on the GPU using Numba CUDA.

    If CUDA is not available or something goes wrong, this function
    will raise an exception and the caller is expected to fall back
    to a CPU backend.
    """
    d_image = cuda.device_array((height, width), dtype=np.int32)

    threads_per_block = (16, 16)
    blocks_x = math.ceil(height / threads_per_block[0])
    blocks_y = math.ceil(width / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)

    mandelbrot_kernel[blocks_per_grid, threads_per_block](
        x_min, x_max, y_min, y_max, max_iter, d_image
    )
    cuda.synchronize()

    return d_image.copy_to_host()


# -------------------------------------------------------------------
# Bounds + coloring
# -------------------------------------------------------------------

def compute_bounds(cx, cy, zoom, aspect, sx=0.5, sy=0.5):
    """
    Compute the complex-plane window from center, zoom, and frame position.

    cx, cy : point to zoom into (on the complex plane)
    zoom   : larger zoom => smaller window => deeper zoom
    aspect : width / height
    sx, sy : where the zoom point appears inside the frame (0..1)
    """
    half_width = 1.5 / zoom
    half_height = half_width / aspect

    full_width = 2.0 * half_width
    full_height = 2.0 * half_height

    x_min = cx - sx * full_width
    x_max = x_min + full_width

    y_min = cy - sy * full_height
    y_max = y_min + full_height

    return x_min, x_max, y_min, y_max


def colorize(image, max_iter, cmap_name="magma"):
    """
    Map iteration counts to an RGB image using a matplotlib colormap.
    """
    img = image.astype(float).copy()

    # points that never escaped (inside the set)
    inside_mask = (img == max_iter)
    img[inside_mask] = 0.0

    # log scale outside the set for smoother gradients
    nonzero_mask = (img > 0)
    img[nonzero_mask] = np.log(img[nonzero_mask])

    vmin = img.min()
    vmax = img.max() if img.max() > vmin else vmin + 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    rgba = cmap(norm(img))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    return rgb
