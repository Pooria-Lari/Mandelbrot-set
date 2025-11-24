import math
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend so Tkinter and matplotlib can work together
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, cuda


# -------------------------------------------------------------------
# Fast Mandelbrot cores (CPU / GPU)
# -------------------------------------------------------------------

@njit(fastmath=True)
def compute_mandelbrot_cpu_single(width, height, max_iter, x_min, x_max, y_min, y_max):
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

            while (zr * zr + zi * zi) <= 4.0 and it < max_iter:
                zr2 = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr2
                it += 1

            image[i, j] = it

    return image


@njit(fastmath=True, parallel=True)
def compute_mandelbrot_cpu_multi(width, height, max_iter, x_min, x_max, y_min, y_max):
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


def compute_mandelbrot_gpu(width, height, max_iter, x_min, x_max, y_min, y_max):
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
# Bounds from center + zoom + position in frame
# -------------------------------------------------------------------

def compute_bounds(cx, cy, zoom, aspect, sx=0.5, sy=0.5):
    """
    Compute the complex-plane window from center, zoom, and frame position.

    cx, cy : point to zoom into (on the complex plane)
    zoom   : larger zoom => smaller window => deeper zoom
    aspect : width / height
    sx, sy : where the zoom point appears inside the frame (0..1)
             sx=0.5, sy=0.5 → center of the frame
             sx=0.3, sy=0.5 → 30% from left, vertically centered
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


# -------------------------------------------------------------------
# Realtime zoom with selectable backend
# -------------------------------------------------------------------

def realtime_zoom(width, height, max_iter,
                  cx, cy, sx, sy,
                  zoom_start, zoom_end, n_frames,
                  cmap_name="magma",
                  backend="cpu_multi"):
    """
    Render a realtime Mandelbrot zoom animation.

    backend:
      "cpu_single"  → Numba, single-core CPU
      "cpu_multi"   → Numba, multi-core CPU (parallel)
      "gpu"         → Numba CUDA (falls back to cpu_multi if unavailable)
    """
    width = int(width)
    height = int(height)

    zoom_start = float(zoom_start)
    zoom_end = float(zoom_end)
    n_frames = int(n_frames)
    cx = float(cx)
    cy = float(cy)
    sx = float(sx)
    sy = float(sy)

    aspect = width / height
    max_iter = int(max_iter)

    zooms = np.geomspace(zoom_start, zoom_end, n_frames)

    backend_mode = backend

    # Warm-up for the chosen backend (JIT compile, CUDA init, etc.)
    print(f"Warm-up backend: {backend_mode}")
    x_min0, x_max0, y_min0, y_max0 = compute_bounds(cx, cy, zooms[0], aspect, sx, sy)

    try:
        if backend_mode == "cpu_single":
            _ = compute_mandelbrot_cpu_single(width, height, max_iter,
                                              x_min0, x_max0, y_min0, y_max0)
        elif backend_mode == "cpu_multi":
            _ = compute_mandelbrot_cpu_multi(width, height, max_iter,
                                             x_min0, x_max0, y_min0, y_max0)
        elif backend_mode == "gpu":
            _ = compute_mandelbrot_gpu(width, height, max_iter,
                                       x_min0, x_max0, y_min0, y_max0)
        else:
            print("Unknown backend, falling back to cpu_multi")
            backend_mode = "cpu_multi"
            _ = compute_mandelbrot_cpu_multi(width, height, max_iter,
                                             x_min0, x_max0, y_min0, y_max0)
    except Exception as e:
        # If GPU or anything else fails here, fall back to CPU multi-core
        print("Error during backend warm-up, switching to CPU multi-core:", e)
        backend_mode = "cpu_multi"
        _ = compute_mandelbrot_cpu_multi(width, height, max_iter,
                                         x_min0, x_max0, y_min0, y_max0)

    # Start animation loop
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = None

    for idx, zoom in enumerate(zooms, start=1):
        x_min, x_max, y_min, y_max = compute_bounds(cx, cy, zoom, aspect, sx, sy)
        print(
            f" frame {idx}/{n_frames} | "
            f"zoom = {zoom:.2f} | x_range = {x_max - x_min:.6f} | backend={backend_mode}"
        )

        # could make max_iter adaptive here; for now we keep it constant
        dyn_max_iter = max_iter

        try:
            if backend_mode == "cpu_single":
                image = compute_mandelbrot_cpu_single(
                    width, height, dyn_max_iter,
                    x_min, x_max, y_min, y_max
                )
            elif backend_mode == "cpu_multi":
                image = compute_mandelbrot_cpu_multi(
                    width, height, dyn_max_iter,
                    x_min, x_max, y_min, y_max
                )
            elif backend_mode == "gpu":
                image = compute_mandelbrot_gpu(
                    width, height, dyn_max_iter,
                    x_min, x_max, y_min, y_max
                )
            else:
                image = compute_mandelbrot_cpu_multi(
                    width, height, dyn_max_iter,
                    x_min, x_max, y_min, y_max
                )
        except Exception as e:
            print("Error in backend during frame, switching to CPU multi-core:", e)
            backend_mode = "cpu_multi"
            image = compute_mandelbrot_cpu_multi(
                width, height, dyn_max_iter,
                x_min, x_max, y_min, y_max
            )

        rgb_frame = colorize(image, dyn_max_iter, cmap_name=cmap_name)

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
        ax.set_title(
            f"Realtime Mandelbrot Zoom  |  zoom = {zoom:.2f}  |  {backend_mode}"
        )

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
    Build a Tkinter control panel for configuring the Mandelbrot zoom
    and selecting the compute backend.
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

    backend_var = tk.StringVar(value="CPU multi-core (Numba)")

    backend_options = [
        "CPU single-core (Numba)",
        "CPU multi-core (Numba)",
        "GPU (CUDA)",
    ]

    backend_map = {
        "CPU single-core (Numba)": "cpu_single",
        "CPU multi-core (Numba)": "cpu_multi",
        "GPU (CUDA)": "gpu",
    }

    def add_row(row, label_text, var):
        ttk.Label(root, text=label_text).grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        entry = ttk.Entry(root, textvariable=var, width=14)
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

    row += 1
    ttk.Label(root, text="Compute backend:").grid(
        row=row, column=0, sticky="w", padx=5, pady=(8, 2)
    )
    backend_combo = ttk.Combobox(
        root,
        textvariable=backend_var,
        values=backend_options,
        state="readonly",
        width=22,
    )
    backend_combo.grid(row=row, column=1, sticky="w", padx=5, pady=(8, 2))
    backend_combo.current(1)  # default: CPU multi-core

    def on_start():
        try:
            backend_label = backend_var.get()
            backend_code = backend_map.get(backend_label, "cpu_multi")

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
                backend=backend_code,
            )
        except Exception as e:
            print("Error in input or rendering:", e)

    row += 1
    ttk.Button(root, text="Start zoom", command=on_start).grid(
        row=row, column=0, columnspan=2, pady=10
    )

    for col in range(2):
        root.grid_columnconfigure(col, weight=1)

    root.mainloop()


if __name__ == "__main__":
    build_ui()
