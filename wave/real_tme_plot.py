import subprocess
import tkinter as tk
from collections import deque

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def create_renderer(
    subprocess_cmd,
    accessor,
    intensity_window=20,
    right_profile_window=20,
    intensity_vmax=1.0,
    wave_vmax=1.0,
):
    class Renderer(tk.Tk):
        def __init__(
            self,
            subprocess_cmd,
            accessor,
            intensity_window,
            right_profile_window,
            intensity_vmax,
            wave_vmax,
        ):
            super().__init__()

            self.title("Wave Propagation")
            self.accessor = accessor
            self.z, self.z_prev = self.accessor()
            self.intensity_window = max(1, int(intensity_window))
            self.right_profile_window = max(1, int(right_profile_window))
            self.intensity_vmax = max(1e-6, float(intensity_vmax))
            self.wave_vmax = max(1e-6, float(wave_vmax))
            self.intensity_history = deque(maxlen=self.intensity_window)
            self.right_profile_history = deque(maxlen=self.right_profile_window)
            self.intensity_sum = np.zeros_like(self.z, dtype=np.float64)
            self.right_profile_sum = np.zeros(self.z.shape[0], dtype=np.float64)
            self.y_coords = np.arange(self.z.shape[0], dtype=np.float64)

            self.figure = Figure(figsize=(8.5, 9), dpi=100)
            grid = self.figure.add_gridspec(
                4,
                2,
                width_ratios=[1.0, 0.18],
                height_ratios=[0.08, 1.0, 0.08, 1.0],
                hspace=0.35,
                wspace=0.08,
            )
            self.cax_wave = self.figure.add_subplot(grid[0, :])
            self.ax = self.figure.add_subplot(grid[1, :])
            self.cax_intensity = self.figure.add_subplot(grid[2, :])
            self.ax_intensity = self.figure.add_subplot(grid[3, 0], sharey=self.ax)
            self.ax_intensity_right = self.figure.add_subplot(grid[3, 1], sharey=self.ax_intensity)
            self.ax.tick_params(labelbottom=False)
            self.ax_intensity_right.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

            self.im = self.ax.imshow(
                self.z,
                cmap="seismic",
                origin="lower",
                vmin=-self.wave_vmax,
                vmax=self.wave_vmax,
            )
            self.colorbar = self.figure.colorbar(self.im, cax=self.cax_wave, orientation="horizontal")
            self.colorbar.set_label("z")
            self.colorbar.ax.xaxis.set_label_position("top")
            self.colorbar.ax.xaxis.set_ticks_position("top")

            current_intensity = np.asarray(self.z, dtype=np.float64) ** 2
            self.intensity_history.append(current_intensity)
            self.intensity_sum[...] = current_intensity
            averaged_intensity = self.intensity_sum / len(self.intensity_history)
            current_right_profile = current_intensity[:, -1].copy()
            self.right_profile_history.append(current_right_profile)
            self.right_profile_sum[...] = current_right_profile
            averaged_right_profile = self.right_profile_sum / len(self.right_profile_history)
            right_profile = averaged_right_profile / max(1e-6, float(np.max(averaged_right_profile)))
            self.intensity_im = self.ax_intensity.imshow(
                averaged_intensity,
                cmap="magma",
                origin="lower",
                vmin=0.0,
                vmax=self.intensity_vmax,
            )
            self.intensity_right_line, = self.ax_intensity_right.plot(
                right_profile,
                self.y_coords,
                color="black",
                linewidth=1.5,
            )
            self.ax_intensity_right.set_xlim(0.0, 1.0)
            self.intensity_colorbar = self.figure.colorbar(
                self.intensity_im,
                cax=self.cax_intensity,
                orientation="horizontal",
            )
            self.intensity_colorbar.set_label("time-averaged intensity")
            self.intensity_colorbar.ax.xaxis.set_label_position("top")
            self.intensity_colorbar.ax.xaxis.set_ticks_position("top")

            self.canvas = FigureCanvasTkAgg(self.figure, self)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvas.draw()

            self.canvas.mpl_connect("button_press_event", self.on_click)
            self.protocol("WM_DELETE_WINDOW", self.on_closing)

            self.process = subprocess.Popen(subprocess_cmd)
            self.after(50, self.update_plot)

        def on_click(self, event):
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                return

            row = int(np.round(event.ydata))
            col = int(np.round(event.xdata))
            self.z[row, col] += 1.0
            self.z_prev[row, col] += 1.0

        def update_plot(self):
            if self.process.poll() is not None:
                self.destroy()
                return

            self.z, self.z_prev = self.accessor()
            self.im.set_data(self.z)

            current_intensity = np.asarray(self.z, dtype=np.float64) ** 2
            if len(self.intensity_history) == self.intensity_window:
                self.intensity_sum -= self.intensity_history[0]
            self.intensity_history.append(current_intensity)
            self.intensity_sum += current_intensity
            averaged_intensity = self.intensity_sum / len(self.intensity_history)
            self.intensity_im.set_data(averaged_intensity)
            current_right_profile = current_intensity[:, -1].copy()
            if len(self.right_profile_history) == self.right_profile_window:
                self.right_profile_sum -= self.right_profile_history[0]
            self.right_profile_history.append(current_right_profile)
            self.right_profile_sum += current_right_profile
            averaged_right_profile = self.right_profile_sum / len(self.right_profile_history)
            right_profile = averaged_right_profile / max(1e-6, float(np.max(averaged_right_profile)))
            self.intensity_right_line.set_data(right_profile, self.y_coords)

            self.colorbar.update_normal(self.im)
            self.canvas.draw_idle()
            self.after(50, self.update_plot)

        def on_closing(self):
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()
            self.destroy()

    return Renderer(
        subprocess_cmd,
        accessor,
        intensity_window,
        right_profile_window,
        intensity_vmax,
        wave_vmax,
    )
