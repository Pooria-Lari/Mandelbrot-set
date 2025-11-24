import tkinter as tk
from tkinter import ttk

from render import realtime_zoom, make_zoom_gif


def build_panel():
    """
    Build and run the Tkinter control panel for the Mandelbrot app.
    """
    root = tk.Tk()
    root.title("Mandelbrot – Realtime & GIF Zoom")

    # core parameters
    width_var = tk.StringVar(value="800")
    height_var = tk.StringVar(value="600")
    max_iter_var = tk.StringVar(value="300")

    cx_var = tk.StringVar(value="-0.75")
    cy_var = tk.StringVar(value="0.0")

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

    # GIF-specific
    gif_filename_var = tk.StringVar(value="mandelbrot_zoom.gif")
    gif_fps_var = tk.StringVar(value="10")

    def add_row(row, label, var, width=14):
        ttk.Label(root, text=label).grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        entry = ttk.Entry(root, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        return entry

    row = 0
    add_row(row, "width:", width_var);   row += 1
    add_row(row, "height:", height_var); row += 1
    add_row(row, "max_iter:", max_iter_var); row += 1

    ttk.Label(root, text="--- Zoom center (cx, cy) ---").grid(
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

    # backend choice
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
    backend_combo.current(1)

    # GIF options
    row += 1
    ttk.Label(root, text="--- GIF export ---").grid(
        row=row, column=0, columnspan=2, pady=(8, 2)
    )
    row += 1
    add_row(row, "GIF filename:", gif_filename_var, width=24); row += 1
    add_row(row, "GIF fps:", gif_fps_var); row += 1

    # Callbacks
    def run_realtime():
        try:
            backend_code = backend_map.get(backend_var.get(), "cpu_multi")

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
            print("Error in realtime render:", e)

    def export_gif():
        try:
            backend_code = backend_map.get(backend_var.get(), "cpu_multi")

            make_zoom_gif(
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
                filename=gif_filename_var.get(),
                fps=gif_fps_var.get(),
            )
        except Exception as e:
            print("Error in GIF export:", e)

    # Buttons
    row += 1
    ttk.Button(root, text="Start realtime zoom", command=run_realtime).grid(
        row=row, column=0, columnspan=2, pady=8
    )
    row += 1
    ttk.Button(root, text="Export zoom GIF", command=export_gif).grid(
        row=row, column=0, columnspan=2, pady=(0, 10)
    )

    for col in range(2):
        root.grid_columnconfigure(col, weight=1)

    root.mainloop()
