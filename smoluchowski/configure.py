import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import numpy as np
import subprocess
from shm_allocator import SharedMemoryAllocator

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Find values at the faces
d1facex = lambda x: np.pad((x[1:, :] + x[:-1, :]) / 2, ((0, 1), (0, 0)), "edge")
d1facey = lambda x: np.pad((x[:, 1:] + x[:, :-1]) / 2, ((0, 0), (0, 1)), "edge")
# Find gradients
d1diffx = lambda x: np.pad((x[1:, :] - x[:-1, :]), ((0, 1), (0, 0)), "edge")
d1diffy = lambda x: np.pad((x[:, 1:] - x[:, :-1]), ((0, 0), (0, 1)), "edge")

# More like upwind scheme when drift-dominated
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
    Initialize shared memory arrays with gradients, face values, and Peclet numbers.
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
    #r_arr = np.ones(c_shape)
    #r_arr[:] = np.arange(0, c_shape[1])
    r_arr = np.arange(0, c_shape[1])

    lambda_n_arr = 1 + 1 / (r_arr*2)
    lambda_n_arr[0] = 2
    lambda_s_arr = 1 - 1 / (r_arr*2)
    lambda_s_arr[0] = 0

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

def compile_cpp(USE_CUDA):
    cpp_file = str(Path(__file__).parent / "smoluchowski.cpp")
    executable = str(Path(__file__).parent / "bin" / "smoluchowski")
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
    try:
        subprocess.run(compile_command, check=True)
        print("Compilation successful.")
        return executable
    except subprocess.CalledProcessError as e:
        print("Compilation failed:", e)
        exit(1)
