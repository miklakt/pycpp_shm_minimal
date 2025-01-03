# %%
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import numpy as np
import subprocess
from shm_allocator import SharedMemoryAllocator

import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# %%
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

allocator = SharedMemoryAllocator(Path(__file__).parent / "shm_layout.json", create_new=True)
allocator.generate_cpp_header(parent_dir / "src/shared_memory_layout.hxx")
allocator.fields["dt"][...] = 0.1
#%%
# Helper functions for operations
d1facex = lambda x: np.pad((x[1:, :] + x[:-1, :]) / 2, ((0, 1), (0, 0)), "edge")
d1facey = lambda x: np.pad((x[:, 1:] + x[:, :-1]) / 2, ((0, 0), (0, 1)), "edge")
d1diffx = lambda x: np.pad((x[1:, :] - x[:-1, :]), ((0, 1), (0, 0)), "edge")
d1diffy = lambda x: np.pad((x[:, 1:] - x[:, :-1]), ((0, 0), (0, 1)), "edge")

def alpha_power_law(Pe):
    alpha = (np.exp(Pe / 2) - 1) / (np.exp(Pe) - 1)
    alpha[np.isclose(Pe, 0)] = 0.5
    return alpha

def redefine_walls_to_faces(W_arr):
    W_faces = np.pad((W_arr[1:, :] + W_arr[:-1, :]), ((0, 1), (0, 0)), "edge")
    W_faces[W_faces > 0] = 1
    return W_faces

def initialize_shared_memory(allocator_, W_arr=None, U_arr=None, D_arr=None):
    """
    Initialize shared memory arrays with gradients, face values, and Peclet numbers
    using the power-law differencing scheme.
    """
    # Infer domain size from shared memory field shapes
    c_shape = allocator_.fields["c"].shape

    # Handle optional inputs
    if W_arr is not None:
        W_arr = redefine_walls_to_faces(np.array(W_arr))
    else:
        W_arr = np.zeros(c_shape, dtype="int8")

    if U_arr is None:
        U_arr = np.zeros(c_shape)
    if D_arr is None:
        D_arr = np.ones(c_shape)

    # Initialize arrays
    r_arr = np.ones(c_shape)
    r_arr[:] = np.arange(0, c_shape[1])

    lambda_n_arr = 1 + 1 / (r_arr*2)
    lambda_n_arr[:, 0] = 2
    lambda_s_arr = 1 - 1 / (r_arr*2)
    lambda_s_arr[:, 0] = 0

    W_not_arr = (W_arr == 0)

    # Gradients and face values
    D_x = d1facex(D_arr) * W_not_arr
    D_y = d1facey(D_arr) * W_not_arr
    dU_x = d1diffx(U_arr) * W_not_arr
    dU_y = d1diffy(U_arr) * W_not_arr

    # Peclet numbers
    Pe_x = -dU_x
    Pe_y = -dU_y

    # Alpha arrays using power-law differencing
    alpha_x = alpha_power_law(Pe_x)
    alpha_y = alpha_power_law(Pe_y)

    # Set shared memory fields
    allocator_.fields["c"][:] = np.zeros(c_shape)
    allocator_.fields["D_x"][:] = D_x
    allocator_.fields["D_y"][:] = D_y
    allocator_.fields["dU_x"][:] = dU_x
    allocator_.fields["dU_y"][:] = dU_y
    allocator_.fields["lambda_n"][:] = lambda_n_arr
    allocator_.fields["lambda_s"][:] = lambda_s_arr
    allocator_.fields["c_next"][:] = np.zeros(c_shape)
    allocator_.fields["alpha_x"][:] = alpha_x
    allocator_.fields["alpha_y"][:] = alpha_y
    allocator_.fields["div_J"][:] = np.zeros(c_shape)

    print("Shared memory fields initialized.")
# %%
cpp_file = str(Path(__file__).parent / "smoluchowski.cpp")
executable = str(Path(__file__).parent / "bin" / "smoluchowski")
#==============
USE_CUDA = True
#==============
if USE_CUDA:
    cpp_file = str(Path(__file__).parent / "smoluchowski.cu")
    compile_command = [
        "nvcc", "-O3", "-std=c++20",
        cpp_file, "-o", executable
    ]
else:
    cpp_file = str(Path(__file__).parent / "smoluchowski.cpp")
    compile_command = [
        "g++", "-O3", "-std=c++20",
        "-fopenmp", 
        "-march=native",
        "-funsafe-math-optimizations",
        cpp_file, "-o", executable
    ]
#compile_command = ["nvcc", "-std=c++20", cpp_file, "-o", executable, "-O3"]
try:
    subprocess.run(compile_command, check=True)
    print("Compilation successful.")
except subprocess.CalledProcessError as e:
    print("Compilation failed:", e)
    exit(1)
# %%
# Example arrays for W_arr, U_arr, and D_arr
z, r = allocator.fields["c"].shape
W_arr = np.zeros((z,r), dtype="int8")
W_arr[z//2-10:z//2+10, 20:-1]=1
U_arr = np.zeros((z,r), dtype="float32")
U_arr[z//2-10:z//2+10, :20]=-1
#%%
initialize_shared_memory(allocator, W_arr=W_arr, U_arr=U_arr)
#%%
def access_shared_memory():
    return allocator.fields["c"].T
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
        app = SharedMemoryPlotApp(subprocess_cmd=[executable, "1000000", "5000"])
    else:
        app = SharedMemoryPlotApp(subprocess_cmd=[executable, "300000"])
    app.mainloop()
    #%%
    print("Subprocess finished. Closing application.")
    allocator.close()
#%%