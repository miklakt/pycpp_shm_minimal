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
3. Python generates `src/shared_memory_layout.hxx` from that layout.
4. C++ reads the generated header and accesses fields via type tags.
5. Solver updates data in shared memory; Python reads and plots it live.

Because the header is generated, both sides always agree on offsets, names, and types.

## Real-Time Monitoring and Plotting

The Python runners start the compiled solver as a subprocess and refresh plots directly from shared memory (Tkinter/QtAgg + Matplotlib).  
This enables live visualization while C++/CUDA keeps running.

## Included Examples

1. **Diffusion equation** time integration  
Path: `diffusion/`

2. **Smoluchowski / drift-diffusion equation** time integration  
Path: `smoluchowski/`

Both examples follow the same pattern:
- define shared-memory layout in JSON,
- generate C++ header from Python,
- compile C++/CUDA executable,
- run solver + live Python plot.

## CUDA Support

CUDA is supported in both examples.

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

## Key Files

- `shm_allocator.py`: shared-memory allocation + C++ header generation
- `src/shared_memory_access.hpp`: typed access to shared-memory fields in C++
- `src/shared_memory_layout.hxx`: auto-generated memory layout header
- `diffusion/*`: diffusion model (CPU and CUDA)
- `smoluchowski/*`: drift-diffusion model (CPU and CUDA)
