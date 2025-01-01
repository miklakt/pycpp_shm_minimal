# %%
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import subprocess
from shm_allocator import SharedMemoryAllocator
import numpy as np
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# %%
# Setup paths
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Initialize Shared Memory Allocator
allocator = SharedMemoryAllocator(Path(__file__).parent / "shm_layout.json", create_new=True)
allocator.generate_cpp_header(parent_dir / "src/shared_memory_layout.hxx")
allocator.fields["dt"][...] = 0.1
# %%
#==============
USE_CUDA = True
#==============

executable = str(Path(__file__).parent / "bin" / "diffusion")
if USE_CUDA:
    cpp_file = str(Path(__file__).parent / "diffusion.cu")
    compile_command = [
        "nvcc", "-O3", "-std=c++20",
        cpp_file, "-o", executable
    ]
else:
    cpp_file = str(Path(__file__).parent / "diffusion.cpp")
    compile_command = [
        "g++", "-O3", "-std=c++20",
        "-fopenmp", 
        "-march=native",
        "-funsafe-math-optimizations",
        cpp_file, "-o", executable
    ]
#%%
try:
    subprocess.run(compile_command, check=True)
    print("Compilation successful.")
except subprocess.CalledProcessError as e:
    print("Compilation failed:", e)
    exit(1)
# %%
# Simulated shared memory access function
def access_shared_memory():
    return allocator.fields["c"].T  # Replace with your shared memory array access

# %%
class SharedMemoryPlotApp(tk.Tk):
    def __init__(self, subprocess_cmd):
        super().__init__()

        self.title("Simple Diffusion Real-Time Visualization")


        # Create Matplotlib figure and axis
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.shared_memory_array = access_shared_memory()

        # Display the initial image
        self.im = self.ax.imshow(
            self.shared_memory_array, 
            cmap="viridis", 
            vmin=0, 
            vmax=1,
            origin='lower'  # Ensure the origin matches the array indexing
        )

        self.colorbar = self.figure.colorbar(self.im, ax=self.ax)

        # Embed the Matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Force an initial draw to ensure the plot is rendered correctly
        self.canvas.draw()

        # Connect the mouse click event
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the subprocess
        self.process = subprocess.Popen(subprocess_cmd)

        # Start updating the plot
        self.update_plot()

    if USE_CUDA:
        on_click = None
    else:
        def on_click(self, event):
            """
            Event handler for mouse clicks on the plot.
            Sets a 10x10 rectangle centered at the click to 1.0 in the shared memory array.
            """
            if event.inaxes != self.ax:
                print("Click outside the plot area.")
                return  # Ignore clicks outside the plot

            # Get the data coordinates from the click
            ydata, xdata = event.xdata, event.ydata

            # Convert to integer indices
            i = int(np.round(ydata))
            j = int(np.round(xdata))

            # Define the size of the rectangle
            rect_size = 10
            half_size = rect_size // 2

            # Get array dimensions
            rows, cols = allocator.fields["c"].shape

            # Calculate the start and end indices, ensuring they are within bounds
            i_start = max(i - half_size, 0)
            i_end = min(i + half_size, rows)
            j_start = max(j - half_size, 0)
            j_end = min(j + half_size, cols)

            # Debug: Print the clicked position and the region to be updated
            #print(f"Clicked at data coordinates ({xdata:.2f}, {ydata:.2f}) -> array indices ({i}, {j}).")
            #print(f"Updating region: rows {i_start}-{i_end}, cols {j_start}-{j_end}")

            # Update the shared memory array
            allocator.fields["c"][i_start:i_end, j_start:j_end] += 10.0

    def update_plot(self):
        # Check if the subprocess has finished
        if self.process.poll() is not None:
            self.destroy()  # Close the Tkinter application
            return

        # Access shared memory data
        try:
            self.shared_memory_array = access_shared_memory()
        except Exception as e:
            print(f"Error accessing shared memory: {e}")
            self.destroy()
            return

        # Update imshow data
        self.im.set_data(self.shared_memory_array)
        # Optionally update color limits if dynamic
        # self.im.set_clim(vmin=self.shared_memory_array.min(), vmax=self.shared_memory_array.max())

        # Redraw the canvas
        self.canvas.draw_idle()  # Use draw_idle for better performance

        # Schedule the next update
        self.after(100, self.update_plot)  # Update every 100ms

    def on_closing(self):
        """
        Handles the window closing event.
        Terminates the subprocess and closes the shared memory allocator.
        """
        print("Closing application...")
        if self.process.poll() is None:
            print("Terminating subprocess...")
            self.process.terminate()
            self.process.wait()
        self.destroy()

# %%
# Run the application
if __name__ == "__main__":
    if USE_CUDA:
        app = SharedMemoryPlotApp(subprocess_cmd=[executable, "300000", "5000"])
    else:
        app = SharedMemoryPlotApp(subprocess_cmd=[executable, "300000"])
    app.mainloop()
    #%%
    print("Subprocess finished. Closing application.")
    allocator.close()
#%%