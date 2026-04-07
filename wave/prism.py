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
SLIT_WALL_X = 32
SLIT_WALL_THICKNESS = 64
SLIT_HALF_HEIGHT = 32
PRISM_CENTER_X = 250
PRISM_ANGLE_DEG = 25.0
PRISM_SIDE = 384
PRISM_MASS = 3
PRISM_CENTER_Y = 330


def make_prism_mass(shape):
    mass = np.ones(shape, dtype=np.float32)
    left = max(0, SLIT_WALL_X - SLIT_WALL_THICKNESS // 2)
    right = min(shape[1], left + SLIT_WALL_THICKNESS)
    mass[:, left:right] = np.inf

    slit = shape[0] // 2
    mass[slit - SLIT_HALF_HEIGHT : slit + SLIT_HALF_HEIGHT + 1, left:right] = 1.0

    center_row = min(PRISM_CENTER_Y, shape[0] - 2)
    center_col = min(PRISM_CENTER_X, shape[1] - 2)
    side = min(PRISM_SIDE, shape[0] - 4, shape[1] - center_col - 4)
    height = side * np.sqrt(3.0) / 2.0

    vertices = np.array(
        [
            [-height / 3.0, -side / 2.0],
            [-height / 3.0, side / 2.0],
            [2.0 * height / 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    theta = np.deg2rad(PRISM_ANGLE_DEG)
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    vertices = vertices @ rotation.T
    vertices[:, 0] += center_col
    vertices[:, 1] += center_row

    yy, xx = np.indices(shape, dtype=np.float64)
    px = xx
    py = yy

    area = (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1]) - (
        vertices[1, 1] - vertices[0, 1]
    ) * (vertices[2, 0] - vertices[0, 0])
    sign = 1.0 if area >= 0.0 else -1.0

    edge0 = sign * ((vertices[1, 0] - vertices[0, 0]) * (py - vertices[0, 1]) - (vertices[1, 1] - vertices[0, 1]) * (px - vertices[0, 0]))
    edge1 = sign * ((vertices[2, 0] - vertices[1, 0]) * (py - vertices[1, 1]) - (vertices[2, 1] - vertices[1, 1]) * (px - vertices[1, 0]))
    edge2 = sign * ((vertices[0, 0] - vertices[2, 0]) * (py - vertices[2, 1]) - (vertices[0, 1] - vertices[2, 1]) * (px - vertices[2, 0]))

    mass[(edge0 >= 0.0) & (edge1 >= 0.0) & (edge2 >= 0.0)] = PRISM_MASS
    return mass


allocator = configure.create_allocator(shape=GRID_SHAPE)

allocator.fields["dt"][...] = 0.01
allocator.fields["spring_k"][...] = 1.0
configure.initialize_shared_memory(
    allocator,
    mass_arr=make_prism_mass(GRID_SHAPE),
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
