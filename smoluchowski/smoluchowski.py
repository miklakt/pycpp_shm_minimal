#%%
import numpy as np
from configure import *
from real_time_plot import create_renderer
#%%
USE_CUDA = False
allocator = SharedMemoryAllocator(Path(__file__).parent / "shm_layout.json", create_new=True)
allocator.generate_cpp_header(parent_dir/"src/shared_memory_layout.hxx")
executable = compile_cpp(USE_CUDA)
#%%
allocator.fields["dt"][...] = 0.1 #timestep

z, r = allocator.fields["c"].shape
W_arr = np.zeros((z,r), dtype="int8") #impermeable walls
W_arr[z//2-10:z//2+10, 20:-1]=1

U_arr = np.zeros((z,r), dtype="float32") #potential
U_arr[z//2-15:z//2+15, :25]=-1
U_arr[z//2-14:z//2+14, :24]=-2
U_arr[z//2-13:z//2+13, :23]=-3
U_arr[z//2-12:z//2+12, :22]=-4
U_arr[z//2-11:z//2+11, :21]=-5
U_arr[z//2-10:z//2+10, :20]=-6
U_arr[z//2-10:z//2+10, 19:21]=-3
U_arr[z//2-9:z//2+9, 18:19]=-2

initialize_shared_memory(allocator, W_arr=W_arr, U_arr=U_arr)
#%%
def access_shared_memory():
    return allocator.fields["c"].T
#%%
if USE_CUDA:
    rendered = create_renderer(subprocess_cmd=[executable, "1000000", "5000"], accessor = access_shared_memory)
else:
    renderer = create_renderer(subprocess_cmd=[executable, "300000"], accessor = access_shared_memory, on_click_command=True)
renderer.mainloop()
#%%
print("Subprocess finished. Closing application.")
allocator.close()
#%%