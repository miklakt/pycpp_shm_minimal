# %%
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import numpy as np
import subprocess
from shm_allocator import SharedMemoryAllocator
# %%
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

allocator = SharedMemoryAllocator(Path(__file__).parent / "shm_layout.json", create_new=True)
allocator.generate_cpp_header(parent_dir / "src/shared_memory_layout.hxx")

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

    # Initialize remaining fields
    # allocator_.fields["grad_c_e"][:] = np.zeros(c_shape)
    # allocator_.fields["grad_c_n"][:] = np.zeros(c_shape)
    # allocator_.fields["J_dif_e"][:] = np.zeros(c_shape)
    # allocator_.fields["J_dif_n"][:] = np.zeros(c_shape)
    # allocator_.fields["J_adv_e"][:] = np.zeros(c_shape)
    # allocator_.fields["J_adv_n"][:] = np.zeros(c_shape)
    # allocator_.fields["J_E"][:] = np.zeros(c_shape)
    # allocator_.fields["J_N"][:] = np.zeros(c_shape)
    # allocator_.fields["div_J"][:] = np.zeros(c_shape)

    print("Shared memory fields initialized.")
# %%
cpp_file = str(Path(__file__).parent / "smoluchowski.cpp")
executable = str(Path(__file__).parent / "bin" / "smoluchowski")
#compile_command = ["g++", "-std=c++23", "-I/usr/include/eigen3", cpp_file, "-o", executable, "-O3"]
compile_command = ["g++", "-fopenmp", "-std=c++23", "-I/usr/include/eigen3", cpp_file, "-o", executable, "-O3"]

try:
    subprocess.run(compile_command, check=True)
    print("Compilation successful.")
except subprocess.CalledProcessError as e:
    print("Compilation failed:", e)
    exit(1)
# %%
# Example arrays for W_arr, U_arr, and D_arr
# W_arr = np.zeros((1000, 1000), dtype="int8")
# U_arr = np.random.rand(1000, 1000)
# D_arr = np.ones((1000, 1000))
#%%
initialize_shared_memory(allocator)
#%%
allocator.fields["dt"][...] = 0.2
#%%
allocator.fields["c"][100:150,100:150] = 1
#%%
plt.imshow(allocator.fields["c"].T, origin = "lower", interpolation = "nearest", aspect =1)
#%%
plt.plot(allocator.fields["c"][:,0])
# Initial conditions
#%%
#%%
import numpy as np
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#%%
# Simulated shared memory access
def access_shared_memory():
    return allocator.fields["c"].T # Replace with your shared memory array access

class SharedMemoryPlotApp(tk.Tk):
    def __init__(self, subprocess_cmd):
        super().__init__()

        self.title("Simple Diffusion Real-Time Visualization")
        self.geometry("1000x1000")

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
            #allocator.close()
            self.destroy()  # Close the Tkinter application
            return
        # Access shared memory data
        self.shared_memory_array = access_shared_memory()

        # Update imshow data
        self.im.set_data(self.shared_memory_array)
        self.im.set_clim(vmin=self.shared_memory_array.min(), vmax=self.shared_memory_array.max())

        # Redraw the canvas
        self.canvas.draw()

        # Schedule the next update
        self.after(100, self.update_plot)  # Update every 100ms


# Run the application
if __name__ == "__main__":
    app = SharedMemoryPlotApp(subprocess_cmd=[executable, "100000"])
    app.mainloop()
    #allocator.close()
    #exit()
#%%