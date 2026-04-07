import json
import sys
from pathlib import Path
import subprocess

import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shm_allocator import SharedMemoryAllocator

script_dir = Path(__file__).resolve().parent
GRID_SHAPE = (256, 512)
LAYOUT_FILE = script_dir / "shm_layout.json"
LAYOUT_HEADER = script_dir / "shared_memory_layout.hxx"


def _layout_spec(shape):
    return {
        "shm_name": "wave_shm",
        "variables": [
            {"name": "dt", "type": "float32"},
            {"name": "timestep", "type": "float32"},
            {"name": "spring_k", "type": "float32"},
            {"name": "oscillator_frequency", "type": "float32"},
        ],
        "arrays": [
            {"name": "z", "type": "float32", "shape": list(shape)},
            {"name": "z_prev", "type": "float32", "shape": list(shape)},
            {"name": "mass", "type": "float32", "shape": list(shape)},
        ],
    }


def create_allocator(create_new=True, shape=GRID_SHAPE):
    LAYOUT_FILE.write_text(json.dumps(_layout_spec(shape), indent=2), encoding="utf-8")
    allocator = SharedMemoryAllocator(LAYOUT_FILE, create_new=create_new)
    allocator.generate_cpp_header(LAYOUT_HEADER)
    return allocator


def initialize_shared_memory(
    allocator,
    mass_arr=None,
    oscillator_frequency=1.0,
):
    shape = allocator.fields["z"].shape

    allocator.fields["z"][:] = 0.0
    allocator.fields["z_prev"][:] = 0.0
    # Reset the simulation clock before the source starts oscillating.
    allocator.fields["timestep"][...] = 0.0
    allocator.fields["oscillator_frequency"][...] = oscillator_frequency

    if mass_arr is None:
        allocator.fields["mass"][:] = 1.0
    else:
        # Use np.inf in the mass field to pin a node in place.
        allocator.fields["mass"][:] = np.asarray(mass_arr, dtype=np.float32)

    print(f"Shared memory initialized for wave grid {shape}.")


def compile_cpp():
    layout_define = f'-DSHM_LAYOUT_HEADER="../{script_dir.name}/shared_memory_layout.hxx"'
    cpp_file = str(script_dir / "wave.cpp")
    executable = str(script_dir / "bin" / "wave")
    Path(executable).parent.mkdir(parents=True, exist_ok=True)

    compile_command = [
        "g++", "-O3", "-std=c++20",
        layout_define,
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
        raise
