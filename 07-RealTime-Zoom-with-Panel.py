import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend so Tkinter and matplotlib work together
import matplotlib.pyplot as plt
from numba import njit
import tkinter as tk
from tkinter import ttk


# -------------------------------------------------------------------
# Fast Mandelbrot core with Numba
# -------------------------------------------------------------------

@njit(fastmath=True)
def compute_mandelbrot_jit(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Compute Mandelbrot escape iterations for a given complex-plane window.
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
            iter_count = 0

            # zr^2 + zi^2 <= 4  <=>  |z| <= 2
            while (zr * zr + zi * zi) <= 4.0 and iter_count < max_iter:
                zr2 = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr2
                iter_count += 1

            image[i, j] = iter_count

    return image


# -------------------------------------------------------------------
# Bounds based on center + zoom + position in frame
# -------------------------------------------------------------------

def compute_bounds(cx, cy, zoom, aspect, sx=0.5, sy=0.5):
    """
    Compute the complex-plane window from a zoom center and how that
    center is placed inside the frame.

    cx, cy : point to zoom into (on the complex plane)
    zoom   : larger zoom => smaller window => deeper zoom
    aspect : width / height of the image
    sx, sy : position of (cx, cy) inside the frame (0..1)
             sx=0.5, sy=0.5 → center of the frame
             sx=0.3, sy=0.5 → 30% from the left, vertically centered
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


# -------------------------------------------------------------------
# Coloring
# -------------------------------------------------------------------

def colorize(image, max_iter, cmap_name="magma"):
    """
    Map raw iteration counts to an RGB image using a matplotlib colormap.
    """
    img = image.astype(float).copy()

    # points that did not escape (inside the set)
    inside_mask = (img == max_iter)
    img[inside_mask] = 0.0

    # log scale for smoother gradients outside the set
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
# Realtime zoom rendering
# -------------------------------------------------------------------

def realtime_zoom(width, height, max_iter,
                  cx, cy, sx, sy,
                  zoom_start, zoom_end, n_frames,
                  cmap_name="magma"):
    """
    Render a realtime Mandelbrot zoom using the Numba-accelerated core
    and parameters provided by the UI.
    """
    width = int(width)
    height = int(height)
    max_iter = int(max_iter)
    zoom_start = float(zoom_start)
    zoom_end = float(zoom_end)
    n_frames = int(n_frames)
    cx = float(cx)
    cy = float(cy)
    sx = float(sx)
    sy = float(sy)

    aspect = width / height
    zooms = np.geomspace(zoom_start, zoom_end, n_frames)

    # JIT warm-up
    print("Compiling Numba kernel...")
    x_min0, x_max0, y_min0, y_max0 = compute_bounds(cx, cy, zooms[0], aspect, sx, sy)
    _ = compute_mandelbrot_jit(width, height, max_iter, x_min0, x_max0, y_min0, y_max0)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = None

    for idx, zoom in enumerate(zooms, start=1):
        x_min, x_max, y_min, y_max = compute_bounds(cx, cy, zoom, aspect, sx, sy)
        print(
            f" frame {idx}/{n_frames} | "
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
        rgb_frame = colorize(image, max_iter, cmap_name=cmap_name)

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

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)

    plt.ioff()
    plt.show()


# -------------------------------------------------------------------
# Tkinter control panel
# -------------------------------------------------------------------

def build_ui():
    """
    Build a simple Tkinter control panel for configuring the zoom
    and launching the realtime Mandelbrot renderer.
    """
    root = tk.Tk()
    root.title("Mandelbrot Realtime Zoom – Settings")

    # text variables with default values
    width_var = tk.StringVar(value="800")
    height_var = tk.StringVar(value="600")
    max_iter_var = tk.StringVar(value="300")

    cx_var = tk.StringVar(value="-0.815")
    cy_var = tk.StringVar(value="0.178")

    # position of the zoom point inside the frame
    sx_var = tk.StringVar(value="0.5")
    sy_var = tk.StringVar(value="0.5")

    zoom_start_var = tk.StringVar(value="1.0")
    zoom_end_var = tk.StringVar(value="300.0")
    n_frames_var = tk.StringVar(value="50")

    cmap_var = tk.StringVar(value="magma")

    # small helper to create a label + entry row
    def add_row(row, label_text, var):
        ttk.Label(root, text=label_text).grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        entry = ttk.Entry(root, textvariable=var, width=12)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        return entry

    row = 0
    add_row(row, "width:", width_var);   row += 1
    add_row(row, "height:", height_var); row += 1
    add_row(row, "max_iter:", max_iter_var); row += 1

    ttk.Label(root, text="--- Zoom point (cx, cy) ---").grid(
        row=row, column=0, columnspan=2, pady=(8, 2)
    )
    row += 1
    add_row(row, "cx (Real):", cx_var); row += 1
    add_row(row, "cy (Imag):", cy_var); row += 1

    ttk.Label(root, text="--- Position in frame (sx, sy) ---").grid(
        row=row, column=0, columnspan=2, pady=(8, 2)
    )
    row += 1
    add_row(row, "sx (0–1):", sx_var); row += 1
    add_row(row, "sy (0–1):", sy_var); row += 1

    ttk.Label(root, text="--- Zoom animation ---").grid(
        row=row, column=0, columnspan=2, pady=(8, 2)
    )
    row += 1
    add_row(row, "zoom_start:", zoom_start_var); row += 1
    add_row(row, "zoom_end:", zoom_end_var);     row += 1
    add_row(row, "frames:", n_frames_var);       row += 1

    ttk.Label(root, text="colormap (e.g. magma, plasma, viridis):").grid(
        row=row, column=0, columnspan=2, pady=(8, 2)
    )
    row += 1
    ttk.Entry(root, textvariable=cmap_var, width=20).grid(
        row=row, column=0, columnspan=2, padx=5, pady=2
    )

    def on_start():
        try:
            realtime_zoom(
                width_var.get(),
                height_var.get(),
                max_iter_var.get(),
                cx_var.get(),
                cy_var.get(),
                sx_var.get(),
                sy_var.get(),
                zoom_start_var.get(),
                zoom_end_var.get(),
                n_frames_var.get(),
                cmap_var.get(),
            )
        except Exception as e:
            print("Error in input or rendering:", e)

    row += 1
    ttk.Button(root, text="Start zoom", command=on_start).grid(
        row=row, column=0, columnspan=2, pady=10
    )

    # make columns stretch a bit
    for col in range(2):
        root.grid_columnconfigure(col, weight=1)

    root.mainloop()


if __name__ == "__main__":
    build_ui()
