# %%
import configure
import numpy as np
from real_tme_plot import create_renderer

# %%
GRID_SHAPE = (512, 512)
OSCILLATOR_FREQUENCY = 0.05
INTENSITY_WINDOW = 100
RIGHT_PROFILE_WINDOW = 400
INTENSITY_VMAX = 0.1
WAVE_VMAX = 1.0
DOUBLE_SLIT_X = 64
SLIT_SEPARATION = 256
SLIT_HALF_HEIGHT = 16


def make_double_slit_mass(shape):
    mass = np.ones(shape, dtype=np.float32)
    mass[:, DOUBLE_SLIT_X] = np.inf

    slit_1 = shape[0] // 2 - SLIT_SEPARATION // 2
    slit_2 = slit_1 + SLIT_SEPARATION
    mass[slit_1 - SLIT_HALF_HEIGHT : slit_1 + SLIT_HALF_HEIGHT + 1, DOUBLE_SLIT_X] = 1.0
    mass[slit_2 - SLIT_HALF_HEIGHT : slit_2 + SLIT_HALF_HEIGHT + 1, DOUBLE_SLIT_X] = 1.0
    return mass


def make_one_slit_mass(shape):
    mass = np.ones(shape, dtype=np.float32)
    mass[:, DOUBLE_SLIT_X] = np.inf

    slit = shape[0] // 2
    mass[slit - SLIT_HALF_HEIGHT : slit + SLIT_HALF_HEIGHT + 1, DOUBLE_SLIT_X] = 1.0
    return mass

allocator = configure.create_allocator(shape=GRID_SHAPE)

allocator.fields["dt"][...] = 0.01
allocator.fields["spring_k"][...] = 1.0
configure.initialize_shared_memory(
    allocator,
    mass_arr=make_double_slit_mass(GRID_SHAPE),
    # mass_arr=make_one_slit_mass(GRID_SHAPE),
    oscillator_frequency=OSCILLATOR_FREQUENCY,
)

executable = configure.compile_cpp()


def access_shared_memory():
    return allocator.fields["z"], allocator.fields["z_prev"]


renderer = create_renderer(
    subprocess_cmd=[executable, "3000000"],
    accessor=access_shared_memory,
    intensity_window=INTENSITY_WINDOW,
    right_profile_window=RIGHT_PROFILE_WINDOW,
    intensity_vmax=INTENSITY_VMAX,
    wave_vmax=WAVE_VMAX,
)
renderer.mainloop()

# %%
print("Subprocess finished. Closing application.")
allocator.close()
