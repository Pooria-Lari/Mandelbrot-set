import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# -------------------------------------------------------------------
# Mandelbrot core
# -------------------------------------------------------------------

def compute_mandelbrot(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Compute raw Mandelbrot iteration counts for a given complex-plane window.

    Returns a 2D integer array of shape (height, width) where each value
    is the number of iterations before escape (or max_iter if it never escaped).
    """
    image = np.zeros((height, width), dtype=int)

    for i in range(height):
        y = y_min + i * (y_max - y_min) / (height - 1)

        for j in range(width):
            x = x_min + j * (x_max - x_min) / (width - 1)
            c = complex(x, y)

            z = 0.0 + 0.0j
            iter_count = 0

            while abs(z) <= 2.0 and iter_count < max_iter:
                z = z * z + c
                iter_count += 1

            image[i, j] = iter_count

        # per-frame progress for this image
        progress = (i + 1) / height * 100.0
        print(f"\r row progress: {progress:.1f}% complete", end="", flush=True)

    print()
    return image


# -------------------------------------------------------------------
# Zoom window computation
# -------------------------------------------------------------------

def compute_bounds(cx, cy, zoom, aspect):
    """
    Given a center point (cx, cy) in the complex plane, a zoom factor,
    and an aspect ratio (width / height), return the bounds:

        x_min, x_max, y_min, y_max

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
    Convert raw iteration counts into an RGB image using a matplotlib colormap.

    - Points that never escaped (== max_iter) are treated as 0 (inside).
    - A logarithmic transform is applied to smooth the color transitions.
    """
    img = image.astype(float).copy()

    # Points inside the set
    inside_mask = (img == max_iter)
    img[inside_mask] = 0.0

    # Log-scale for smoother gradients outside the set
    nonzero_mask = (img > 0)
    img[nonzero_mask] = np.log(img[nonzero_mask])

    # Normalize for colormap
    vmin = img.min()
    vmax = img.max() if img.max() > vmin else vmin + 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    rgba = cmap(norm(img))                    # values in [0, 1], with alpha
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)  # drop alpha, go to [0, 255]

    return rgb


# -------------------------------------------------------------------
# Zoom animation
# -------------------------------------------------------------------

def make_zoom_animation(filename="mandelbrot_zoom.gif"):
    """
    Generate a zoom-in Mandelbrot GIF animation.
    """
    width = 800
    height = 600
    max_iter = 300

    aspect = width / height

    # zoom center (pick a visually interesting point)
    cx = -0.75
    cy = 0.0

    # zoom range
    zoom_start = 1.0    # wide overview
    zoom_end = 60.0     # deeper zoom
    n_frames = 40       # number of frames

    # use geometric spacing so zoom accelerates visually
    zooms = np.geomspace(zoom_start, zoom_end, n_frames)

    frames = []

    for idx, zoom in enumerate(zooms, start=1):
        print(f"\nFrame {idx}/{n_frames}  |  zoom = {zoom:.2f}")

        x_min, x_max, y_min, y_max = compute_bounds(cx, cy, zoom, aspect)

        image = compute_mandelbrot(
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

    # write GIF
    imageio.mimsave(filename, frames, fps=10)
    print(f"\nAnimation saved to: {filename}")


if __name__ == "__main__":
    make_zoom_animation()
