#%%
import json
from pathlib import Path
import numpy as np
from multiprocessing import shared_memory

def spec_to_dtype(type_str: str) -> np.dtype:
    """
    Convert a string describing a numeric data type into a NumPy dtype.
    Extend or modify this dictionary to support additional types.
    """
    dtype_map = {
        "int8":    np.int8,
        "uint8":   np.uint8,
        "int16":   np.int16,
        "uint16":  np.uint16,
        "int32":   np.int32,
        "uint32":  np.uint32,
        "int64":   np.int64,
        "uint64":  np.uint64,
        "float32": np.float32,
        "float64": np.float64,
    }
    try:
        return np.dtype(dtype_map[type_str])
    except KeyError:
        raise ValueError(f"Unsupported or unknown type string: '{type_str}'")

class SharedMemoryAllocator:
    """
    Manages a single shared memory segment based on a JSON specification
    describing scalar variables and multi-dimensional arrays.

    If create_new is True, a new segment is created/unlinked.
    Otherwise, attempts to connect to an existing segment by name.
    """

    def __init__(self, spec_file: str, create_new: bool = True):
        """
        :param spec_file: Path to the JSON specification.
        :param create_new: If True, create or recreate the shared memory.
                           If False, connect to an existing shared memory segment.
        """
        self.spec_file = spec_file
        self.create_new = create_new
        self.shm = None
        self.fields = {}  # Will hold the actual NumPy arrays (keyed by field name)
        self._parse_and_allocate_or_connect()

    def _parse_spec(self):
        """
        Loads the JSON spec, computes total size, and returns:
          - shm_name: the name of the shared memory segment
          - total_size: total byte size needed
          - layout_info: list of { 'name', 'dtype', 'shape', 'offset' }
        """
        spec = json.loads(Path(self.spec_file).read_text())
        layout_info = []
        current_offset = 0

        # Variables (scalars)
        for var in spec.get("variables", []):
            dt = spec_to_dtype(var["type"])
            size_bytes = dt.itemsize
            layout_info.append({
                "name": var["name"],
                "dtype": dt,
                "shape": None,  # indicates scalar
                "offset": current_offset
            })
            current_offset += size_bytes

        # Arrays (fields)
        for arr in spec.get("arrays", []):
            dt = spec_to_dtype(arr["type"])
            shape = arr["shape"]
            num_elems = np.prod(shape)
            size_bytes = dt.itemsize * num_elems
            layout_info.append({
                "name": arr["name"],
                "dtype": dt,
                "shape": shape,
                "offset": current_offset
            })
            current_offset += size_bytes

        return spec["shm_name"], current_offset, layout_info

    def _parse_and_allocate_or_connect(self):
        # 1) Parse the JSON spec
        shm_name, total_size, layout_info = self._parse_spec()

        # 2) Create or connect to the shared memory
        if self.create_new:
            # Re-create the shared memory block to ensure it's fresh
            # Attempt to clean up any existing segment with that name
            try:
                existing = shared_memory.SharedMemory(name=shm_name)
                print(f"[SharedMemoryAllocator] WARNING: A shared memory segment "
                        f"'{shm_name}' already exists. Unlinking it to create a new one.")
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                pass

            self.shm = shared_memory.SharedMemory(
                create=True,
                size=total_size,
                name=shm_name
            )
            print(f"[SharedMemoryAllocator] Created new shared memory '{shm_name}' ({total_size} bytes).")
        else:
            # Connect to an existing shared memory
            # This will raise FileNotFoundError if it doesn't exist
            self.shm = shared_memory.SharedMemory(
                create=False,
                size=total_size,  # technically you can pass size=0 for attach-only
                name=shm_name
            )
            print(f"[SharedMemoryAllocator] Connected to existing shared memory '{shm_name}' ({total_size} bytes).")

        # 3) Create NumPy arrays mapped onto the shared memory buffer
        for item in layout_info:
            offset = item["offset"]
            dtype = item["dtype"]
            print(f"Offset: {offset}")
            if item["shape"] is None:
                # Scalar => store as shape ()
                arr = np.ndarray((), dtype=dtype, buffer=self.shm.buf, offset=offset)
            else:
                arr = np.ndarray(item["shape"], dtype=dtype, buffer=self.shm.buf, offset=offset)
            
            # Store in a dictionary for easy access
            self.fields[item["name"]] = arr

    def initialize_data(self, inits: dict):
        """
        Convenience method to set initial values for some/all fields.
        Example:
            allocator.initialize_data({
                "myint": 42,
                "myfloat": 3.14,
                "myarr": np.array([[1,2,3],[4,5,6]], dtype=np.float32)
            })
        """
        for key, value in inits.items():
            if key not in self.fields:
                raise KeyError(f"Field '{key}' not found in shared memory layout.")
            arr = self.fields[key]
            if arr.ndim == 0:  
                # 0D scalar: assign using arr[...] or arr[()]
                arr[...] = value
            else:
                # Multi-dimensional array
                arr[...] = value

    def close(self, unlink=True):
        """
        Closes the shared memory block. Optionally unlinks (removes) it.
        After this call, arrays are no longer valid.
        """
        if self.shm is not None:
            print("[SharedMemoryAllocator] Closing shared memory...")
            self.shm.close()
            if unlink and self.create_new:
                # Only unlink if we originally created it. 
                # Typically you don't want to unlink if you only "attached".
                self.shm.unlink()
                print("[SharedMemoryAllocator] Unlinked the shared memory.")
            self.shm = None

    def __del__(self):
        """
        Destructor: ensure resources are released if not closed manually.
        """
        self.close(unlink=False)
#%%
allocator = SharedMemoryAllocator("shm_layout.json", create_new=True)
#%%
print("Fields in new shared memory:", list(allocator.fields.keys()))
print("myint =", allocator.fields["myint"])
print("myarr =\n", allocator.fields["myarr"])
# Do not close/unlink yet if we want another process to attach...
# allocator_new.close()