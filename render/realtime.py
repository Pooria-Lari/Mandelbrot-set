import matplotlib
matplotlib.use("TkAgg")  # use TkAgg backend for interactive window

import matplotlib.pyplot as plt
import numpy as np

from core import (
    compute_mandelbrot_cpu_single,
    compute_mandelbrot_cpu_multi,
    compute_mandelbrot_gpu,
    compute_bounds,
    colorize,
)


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

    # Warm-up: JIT compile / initialize chosen backend
    print(f"Warm-up backend: {backend_mode}")
    x_min0, x_max0, y_min0, y_max0 = compute_bounds(
        cx, cy, zooms[0], aspect, sx, sy
    )

    try:
        if backend_mode == "cpu_single":
            _ = compute_mandelbrot_cpu_single(
                width, height, max_iter, x_min0, x_max0, y_min0, y_max0
            )
        elif backend_mode == "cpu_multi":
            _ = compute_mandelbrot_cpu_multi(
                width, height, max_iter, x_min0, x_max0, y_min0, y_max0
            )
        elif backend_mode == "gpu":
            _ = compute_mandelbrot_gpu(
                width, height, max_iter, x_min0, x_max0, y_min0, y_max0
            )
        else:
            print("Unknown backend, falling back to cpu_multi")
            backend_mode = "cpu_multi"
            _ = compute_mandelbrot_cpu_multi(
                width, height, max_iter, x_min0, x_max0, y_min0, y_max0
            )
    except Exception as e:
        print("Backend warm-up failed, switching to CPU multi-core:", e)
        backend_mode = "cpu_multi"
        _ = compute_mandelbrot_cpu_multi(
            width, height, max_iter, x_min0, x_max0, y_min0, y_max0
        )

    # Animation loop
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = None

    for idx, zoom in enumerate(zooms, start=1):
        x_min, x_max, y_min, y_max = compute_bounds(
            cx, cy, zoom, aspect, sx, sy
        )
        print(
            f" frame {idx}/{n_frames} | "
            f"zoom = {zoom:.2f} | x_range = {x_max - x_min:.6f} | backend={backend_mode}"
        )

        dyn_max_iter = max_iter  # could be made adaptive if you like

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
            print("Error during frame, switching to CPU multi-core:", e)
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
