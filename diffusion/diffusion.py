# %%
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import subprocess
from shm_allocator import SharedMemoryAllocator
# %%
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

allocator = SharedMemoryAllocator(Path(__file__).parent / "shm_layout.json", create_new=True)
allocator.generate_cpp_header(parent_dir / "src/shared_memory_layout.hxx")

# %%
cpp_file = str(Path(__file__).parent / "diffusion.cpp")
executable = str(Path(__file__).parent / "bin" / "diffusion")
compile_command = ["g++", "-std=c++23", "-I/usr/include/eigen3", cpp_file, "-o", executable, "-O3"]

try:
    subprocess.run(compile_command, check=True)
    print("Compilation successful.")
except subprocess.CalledProcessError as e:
    print("Compilation failed:", e)
    exit(1)

# %%
# Initial conditions
allocator.fields["dt"][...] = 0.1
allocator.fields["c"][450:550, 450:550] = 100.0
#%%
import numpy as np
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#%%
# Simulated shared memory access
def access_shared_memory():
    return allocator.fields["c"] # Replace with your shared memory array access

class SharedMemoryPlotApp(tk.Tk):
    def __init__(self, subprocess_cmd):
        super().__init__()

        self.title("Simple Diffusion Real-Time Visualization")
        self.geometry("800x800")

        # Create Matplotlib figure and axis
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.shared_memory_array = access_shared_memory()
        self.im = self.ax.imshow(self.shared_memory_array, cmap="viridis")
        self.colorbar = self.figure.colorbar(self.im, ax=self.ax)

        # Embed the Matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Start the subprocess
        self.process = subprocess.Popen(subprocess_cmd)
        # Start updating the plot
        self.update_plot()

    def update_plot(self):
        # Check if the subprocess has finished
        if self.process.poll() is not None:
            print("Subprocess finished. Closing application.")
            allocator.close()
            self.destroy()  # Close the Tkinter application
            return
        # Access shared memory data
        self.shared_memory_array = access_shared_memory()

        # Update imshow data
        self.im.set_data(self.shared_memory_array)
        #self.im.set_clim(vmin=self.shared_memory_array.min(), vmax=self.shared_memory_array.max())

        # Redraw the canvas
        self.canvas.draw()

        # Schedule the next update
        self.after(100, self.update_plot)  # Update every 100ms


# Run the application
if __name__ == "__main__":
    app = SharedMemoryPlotApp(subprocess_cmd=[executable, "10000"])
    app.mainloop()
    exit()