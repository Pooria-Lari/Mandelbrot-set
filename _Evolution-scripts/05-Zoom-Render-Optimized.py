import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from numba import njit, prange


# -------------------------------------------------------------------
# Mandelbrot core (Numba-accelerated, parallel)
# -------------------------------------------------------------------

@njit(fastmath=True, parallel=True)
def compute_mandelbrot_jit(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Compute Mandelbrot escape iterations for a given complex-plane window.

    This version uses Numba with prange to parallelize over image rows.
    """
    image = np.zeros((height, width), dtype=np.int32)

    # precompute pixel spacing on x and y axes
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
# View window / zoom utilities
# -------------------------------------------------------------------

def compute_bounds(cx, cy, zoom, aspect):
    """
    Given a center point (cx, cy), a zoom factor and aspect ratio,
    return the complex-plane window [x_min, x_max, y_min, y_max].

    Larger zoom => smaller window => deeper zoom.
    """
    half_width = 1.5 / zoom
    half_height = half_width / aspect

    x_min = cx - half_width
    x_max = cx + half_width
    y_min = cy - half_height
    y_max = cy + half_height

    return x_min, x_max, y_min, y_max


# -------------------------------------------------------------------
# Coloring
# -------------------------------------------------------------------

def colorize(image, max_iter, cmap_name="magma"):
    """
    Convert raw iteration counts into an RGB frame using a matplotlib colormap.

    - Points that never escaped (== max_iter) are treated as 0 (inside the set).
    - A log transform is applied to smooth the gradient outside the set.
    """
    img = image.astype(float).copy()

    # points inside the set
    inside_mask = (img == max_iter)
    img[inside_mask] = 0.0

    # log scaling for smoother gradients
    nonzero_mask = (img > 0)
    img[nonzero_mask] = np.log(img[nonzero_mask])

    # normalize for colormap
    vmin = img.min()
    vmax = img.max() if img.max() > vmin else vmin + 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    rgba = cmap(norm(img))                      # in [0, 1], with alpha
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)  # drop alpha, 0–255

    return rgb


# -------------------------------------------------------------------
# Zoom animation
# -------------------------------------------------------------------

def make_zoom_animation(filename="mandelbrot_zoom.gif"):
    """
    Render a Mandelbrot zoom as an animated GIF using the JIT-accelerated core.
    """
    width = 800
    height = 600
    max_iter = 300

    aspect = width / height

    # zoom center (a classic interesting spot on the Mandelbrot set)
    cx = -0.75
    cy = 0.0

    # zoom range
    zoom_start = 1.0    # wide view
    zoom_end = 60.0     # deeper zoom
    n_frames = 40       # number of frames in the GIF

    # geometric zoom progression (visually smoother acceleration)
    zooms = np.geomspace(zoom_start, zoom_end, n_frames)

    frames = []

    for idx, zoom in enumerate(zooms, start=1):
        print(f"\nFrame {idx}/{n_frames}  |  zoom = {zoom:.2f}")

        x_min, x_max, y_min, y_max = compute_bounds(cx, cy, zoom, aspect)

        # JIT-accelerated Mandelbrot computation
        image = compute_mandelbrot_jit(
            width,
            height,
            max_iter,
            x_min,
            x_max,
            y_min,
            y_max,
        )

        rgb_frame = colorize(image, max_iter, cmap_name="magma")
        frames.append(rgb_frame)

    # write animated GIF
    imageio.mimsave(filename, frames, fps=10)
    print(f"\nAnimation saved to: {filename}")


if __name__ == "__main__":
    make_zoom_animation()
