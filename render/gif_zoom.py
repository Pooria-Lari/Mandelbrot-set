import matplotlib
matplotlib.use("TkAgg")  # safe to use TkAgg here as well

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

from core import (
    compute_mandelbrot_cpu_single,
    compute_mandelbrot_cpu_multi,
    compute_mandelbrot_gpu,
    compute_bounds,
    colorize,
)


def make_zoom_gif(width, height, max_iter,
                  cx, cy, sx, sy,
                  zoom_start, zoom_end, n_frames,
                  cmap_name="magma",
                  backend="cpu_multi",
                  filename="mandelbrot_zoom.gif",
                  fps=10):
    """
    Render a Mandelbrot zoom into a GIF file.
    Uses the same backends as the realtime renderer.
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
    fps = int(fps)

    aspect = width / height
    zooms = np.geomspace(zoom_start, zoom_end, n_frames)
    backend_mode = backend

    # Warm-up for chosen backend
    print(f"Warm-up backend for GIF: {backend_mode}")
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

    frames = []

    for idx, zoom in enumerate(zooms, start=1):
        x_min, x_max, y_min, y_max = compute_bounds(
            cx, cy, zoom, aspect, sx, sy
        )
        print(
            f" GIF frame {idx}/{n_frames} | "
            f"zoom = {zoom:.2f} | x_range = {x_max - x_min:.6f} | backend={backend_mode}"
        )

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
            print("Error in backend during GIF frame, switching to CPU multi-core:", e)
            backend_mode = "cpu_multi"
            image = compute_mandelbrot_cpu_multi(
                width, height, dyn_max_iter,
                x_min, x_max, y_min, y_max
            )

        rgb_frame = colorize(image, dyn_max_iter, cmap_name=cmap_name)
        frames.append(rgb_frame)

    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved to: {filename}")
