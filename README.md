# pycpp_shm_minimal

Minimal project showing how to run numerical simulation in C++/CUDA while exposing results to Python in real time through shared memory.

## Goal

The goal is to do the heavy calculations in C++ (or CUDA), and use Python for:
- initializing inputs,
- starting the solver process,
- real-time monitoring,
- and plotting of the computed fields.

## Core Idea

This project uses a shared-memory contract that is defined once and consumed by both Python and C++.

1. A JSON layout defines variables/arrays and their types.
2. Python (`shm_allocator.py`) allocates shared memory and maps NumPy views.
3. Python generates a `shared_memory_layout.hxx` header.
4. C++ reads that generated header (selected at compile time) and accesses fields via type tags.
5. The compiled C++ code updates shared memory; Python reads and plots it live (or vice versa).

Because the header is generated, both sides always agree on offsets, names, and types.

## Real-Time Monitoring and Plotting

The Python runners start the compiled solver as a subprocess and refresh plots directly from shared memory (Tkinter/QtAgg + Matplotlib).  
This enables live visualization while C++/CUDA keeps running.

## Included Examples

1. **Diffusion equation** time integration  
Path: `diffusion/`

2. **Smoluchowski / drift-diffusion equation** time integration  
Path: `smoluchowski/`

3. **Wave propagation in 2D cellular automata**  
Path: `wave/`

4. **C++ shared-memory access examples**  
Path: `cpp_examples/`

The simulation examples follow the same pattern:
- define shared-memory layout in JSON,
- generate C++ header from Python,
- compile C++/CUDA executable,
- run solver + live Python plot.

## CUDA Support

CUDA is supported in both simulation examples.

- `diffusion/diffusion.py` has `USE_CUDA = True/False`
- `smoluchowski/smoluchowski.py` has `USE_CUDA = True/False`

When enabled, scripts compile `.cu` files with `nvcc`.  
When disabled, scripts compile `.cpp` files with `g++`.

## Quick Start

Run from repository root.

### Requirements

- Python 3
- `numpy`, `matplotlib`
- Tkinter (for interactive plotting windows)
- C++ compiler with C++20 support (`g++`)
- Optional: CUDA toolkit (`nvcc`) for GPU mode

### Run diffusion example

```bash
python3 diffusion/diffusion.py
```

### Run Smoluchowski example

```bash
python3 smoluchowski/smoluchowski.py
```

### Run wave double slit example

```bash
python3 wave/double_slit.py
```

### Run wave prism example

```bash
python3 wave/prism.py
```

### Run C++ access examples

```bash
# Terminal 1 (keep running)
python3 cpp_examples/create_shared_memory.py

# Terminal 2
mkdir -p cpp_examples/bin
g++ -std=c++20 -O3 -DSHM_DISABLE_FIELD_ALIASES -DSHM_LAYOUT_HEADER=\"../cpp_examples/shared_memory_layout.hxx\" cpp_examples/access_by_name_example.cpp -o cpp_examples/bin/access_by_name_example
g++ -std=c++23 -O3 -I/usr/include/eigen3 -DSHM_DISABLE_FIELD_ALIASES -DSHM_LAYOUT_HEADER=\"../cpp_examples/shared_memory_layout.hxx\" cpp_examples/eigen_map_example.cpp -o cpp_examples/bin/eigen_map_example
./cpp_examples/bin/access_by_name_example
```

## Key Files

- `shm_allocator.py`: shared-memory allocation + C++ header generation
- `cpp_examples/create_shared_memory.py`: creates C++-example shared memory + header
- `src/shared_memory_access.hpp`: typed access to shared-memory fields in C++
- `cpp_examples/access_by_name_example.cpp`: basic C++ field access by tag/name
- `cpp_examples/eigen_map_example.cpp`: Eigen mapping example on shared-memory fields
- `cpp_examples/eigen_map.hpp`: Eigen helper utilities for shared-memory arrays
- `cpp_examples/shm_layout_example.json`: layout spec for C++ access examples
- `diffusion/*`: diffusion model
- `diffusion/shm_layout.json`: layout spec for diffusion example
- `smoluchowski/*`: drift-diffusion model
- `smoluchowski/shm_layout.json`: layout spec for smoluchowski example
