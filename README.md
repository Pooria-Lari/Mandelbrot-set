# Mandelbrot Set Explorer

Interactive Mandelbrot set explorer with CPU / multi-core / GPU backends, real-time zoom, and GIF export — all wrapped in a simple Tkinter control panel.

https://github.com/Pooria-Lari/Mandelbrot-set/


## Features

- 🔍 **Real-time Mandelbrot zoom**
  - Adjustable center (`cx`, `cy`)
  - Control where the zoom point appears in the frame (`sx`, `sy`)
  - Geometric zoom progression (smooth zoom-in)

- ⚙️ **Multiple compute backends**
  - `CPU single-core` (Numba)
  - `CPU multi-core` (Numba + parallel `prange`)
  - `GPU (CUDA)` via Numba (optional, if you have a supported NVIDIA GPU)

- 🎨 **Flexible coloring**
  - Log-scaled iteration counts
  - Any matplotlib colormap (`magma`, `plasma`, `viridis`, …)

- 🎞 **GIF export**
  - Render zoom animations directly to a `.gif` file
  - Adjustable resolution, frame count, zoom range, and FPS

- 🖱 **Simple GUI**
  - Tkinter-based control panel
  - No command-line arguments required


## Project structure

```text
Mandelbrot-set/
├─ _Evolution-scripts/
│   ├─ 00-Base.py
│   ├─ 01-ColorVer.py
│   ├─ 02-JIT-Ver.py
│   ├─ 03-JIT-Ver2.py
│   ├─ 04-Zoom-Render.py
│   ├─ 05-Zoom-Render-Optimized.py
│   ├─ 06-RealTime-Zoom-Render.py
│   ├─ 07-RealTime-Zoom-with-Panel.py
│   └─ 08-Optimization.py
│
├─ core/
│   └─ mandelbrot_core.py
│
├─ render/
│   ├─ realtime.py
│   └─ gif_zoom.py
│
├─ ui/
│   └─ panel.py
│
├─ main.py
├─ requirements.txt
└─ README.md


