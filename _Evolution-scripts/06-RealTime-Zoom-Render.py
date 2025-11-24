import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# -------------------------------------------------------------------
# Fast Mandelbrot core with Numba
# -------------------------------------------------------------------

@njit(fastmath=True)
def compute_mandelbrot_jit(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Compute Mandelbrot escape iterations for a given complex-plane window.

    This version is JIT-compiled with Numba for a big speed-up
    compared to pure Python.
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


# -------------------------------------------------------------------
# Bounds from center + zoom
# -------------------------------------------------------------------

def compute_bounds(cx, cy, zoom, aspect):
    """
    Given a center point (cx, cy), zoom factor, and aspect ratio (w/h),
    compute the complex-plane window [x_min, x_max, y_min, y_max].

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
    Map raw iteration counts to an RGB image using a matplotlib colormap.
    Uses log scaling for smoother gradients.
    """
    img = image.astype(float).copy()

    # points that did not escape (inside the set)
    inside_mask = (img == max_iter)
    img[inside_mask] = 0.0

    # log scale outside the set
    nonzero_mask = (img > 0)
    img[nonzero_mask] = np.log(img[nonzero_mask])

    vmin = img.min()
    vmax = img.max() if img.max() > vmin else vmin + 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    rgba = cmap(norm(img))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    return rgb


# -------------------------------------------------------------------
# Realtime zoom animation
# -------------------------------------------------------------------

def realtime_zoom():
    """
    Show a realtime Mandelbrot zoom using the Numba-accelerated core.
    """
    width = 800
    height = 600
    max_iter = 200
    aspect = width / height

    # initial zoom center
    cx = -0.75
    cy = 0.0

    # make the zoom visually obvious
    zoom_start = 1.0
    zoom_end = 300.0
    n_frames = 50

    zooms = np.geomspace(zoom_start, zoom_end, n_frames)

    # --- JIT warm-up (first compilation can be slow) ---
    print("Compiling Numba kernel for the first time...")
    x_min0, x_max0, y_min0, y_max0 = compute_bounds(cx, cy, zooms[0], aspect)
    _ = compute_mandelbrot_jit(width, height, max_iter, x_min0, x_max0, y_min0, y_max0)

    # --- interactive matplotlib mode ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = None

    for idx, zoom in enumerate(zooms, start=1):
        x_min, x_max, y_min, y_max = compute_bounds(cx, cy, zoom, aspect)

        # numeric sanity check in the console
        print(
            f"frame {idx}/{n_frames} | "
            f"zoom = {zoom:.2f} | x_range = {x_max - x_min:.6f}"
        )

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

        if im is None:
            im = ax.imshow(
                rgb_frame,
                extent=(x_min, x_max, y_min, y_max),
                origin="lower",
                animated=True,
            )
            ax.set_xlabel("Real axis")
            ax.set_ylabel("Imaginary axis")
        else:
            im.set_data(rgb_frame)
            im.set_extent((x_min, x_max, y_min, y_max))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Realtime Mandelbrot Zoom  |  zoom = {zoom:.2f}")

        # force GUI update in some environments
        fig.canvas.draw()
        fig.canvas.flush_events()

        # short pause so the eye can see the change
        plt.pause(0.05)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    realtime_zoom()
