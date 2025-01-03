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
        self.layout_info = []   # list of { 'name', 'dtype', 'shape', 'offset' }
        self.total_size = 0  # Initialize total size
        self._parse_and_allocate_or_connect()

    def _parse_spec(self):
        """
        Loads the JSON spec, computes total size, and returns:
          - shm_name: the name of the shared memory segment
          - total_size: total byte size needed
        """
        spec = json.loads(Path(self.spec_file).read_text())
        current_offset = 0

        # Variables (scalars)
        for var in spec.get("variables", []):
            dt = spec_to_dtype(var["type"])
            size_bytes = dt.itemsize
            self.layout_info.append({
                "name": var["name"],
                "dtype": dt,
                "shape": (),  # indicates scalar
                "offset": current_offset
            })
            current_offset += size_bytes
            self.total_size += size_bytes

        # Arrays (fields)
        for arr in spec.get("arrays", []):
            dt = spec_to_dtype(arr["type"])
            shape = arr["shape"]
            num_elems = np.prod(shape)
            size_bytes = dt.itemsize * num_elems
            self.layout_info.append({
                "name": arr["name"],
                "dtype": dt,
                "shape": shape,
                "offset": current_offset
            })
            current_offset += size_bytes
            self.total_size += size_bytes

        return spec["shm_name"]

    def _parse_and_allocate_or_connect(self):
        # 1) Parse the JSON spec and compute total_size
        shm_name = self._parse_spec()

        # 2) Create or connect to the shared memory
        if self.create_new:
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
                size=self.total_size,
                name=shm_name
            )
            print(f"[SharedMemoryAllocator] Created new shared memory '{shm_name}' ({self.total_size} bytes).")
        else:
            self.shm = shared_memory.SharedMemory(
                create=False,
                size=self.total_size,  # Pass total_size instead of using lseek
                name=shm_name
            )
            print(f"[SharedMemoryAllocator] Connected to existing shared memory '{shm_name}' ({self.total_size} bytes).")

        # 3) Create NumPy arrays mapped onto the shared memory buffer
        for item in self.layout_info:
            offset = item["offset"]
            dtype = item["dtype"]
            print(f"Offset: {offset}")
            arr = np.ndarray(item["shape"], dtype=dtype, buffer=self.shm.buf, offset=offset)
            self.fields[item["name"]] = arr

    def initialize_data(self, inits: dict):
        """
        Convenience method to set initial values for some/all fields.
        """
        for key, value in inits.items():
            if key not in self.fields:
                raise KeyError(f"Field '{key}' not found in shared memory layout.")
            arr = self.fields[key]
            if arr.ndim == 0:  
                arr[...] = value
            else:
                arr[...] = value

    def close(self, unlink=True):
        """
        Closes the shared memory block. Optionally unlinks (removes) it.
        """
        if self.shm is not None:
            print("[SharedMemoryAllocator] Closing shared memory...")
            self.shm.close()
            if unlink and self.create_new:
                self.shm.unlink()
                print("[SharedMemoryAllocator] Unlinked the shared memory.")
            self.shm = None

    def __del__(self):
        """
        Destructor: ensure resources are released if not closed manually.
        """
        self.close(unlink=False)

    def _numpy_dtype_to_cpp(self, dtype: str) -> str:
        """
        Map NumPy dtype strings to C++ types.
        Extend this method to support more types as needed.
        """
        mapping = {
            np.dtype("int8")    : "int8_t",
            np.dtype("uint8")   : "uint8_t",
            np.dtype("int16")   : "int16_t",
            np.dtype("uint16")  : "uint16_t",
            np.dtype("int32")   : "int32_t",
            np.dtype("uint32")  : "uint32_t",
            np.dtype("int64")   : "int64_t",
            np.dtype("uint64")  : "uint64_t",
            np.dtype("float32") : "float",
            np.dtype("float64") : "double",
        }
        return mapping.get(dtype, None)

    def generate_cpp_header(self, output_file: str = "shared_memory_layout"):
        """
        Generates a C++ header that defines compile-time offsets for each field
        found in `self.layout_info`, along with the total size of the shared memory.
        """
        lines = []
        lines.append("// This file is AUTO-GENERATED from the SharedMemoryAllocator class. Do not edit manually.")
        lines.append("#pragma once")
        lines.append("#include <cstddef>")
        lines.append("")
        spec = json.loads(Path(self.spec_file).read_text())
        lines.append(f'inline constexpr const char* SHM_NAME = "{spec["shm_name"]}";')
        lines.append(f'inline constexpr std::size_t SHM_SIZE = {self.total_size};' + "// Bytes")
        lines.append("")
        lines.append("namespace SharedMemoryLayout {")
        lines.append("")

        # 1) For each field name, generate a unique tag
        for item in self.layout_info:
            name = item["name"]
            tag_name = f"{name}_tag"
            lines.append(f"    struct {tag_name} {{}};")

        lines.append("")
        # 2) Primary template for field_info<Tag>
        lines.append("    template <typename Tag>")
        lines.append("    struct field_info; // forward declaration (no definition)")
        lines.append("")

        # 3) Specializations
        for item in self.layout_info:
            name = item["name"]
            offset = item["offset"]
            tag_name = f"{name}_tag"
            shape = item["shape"]
            dtype = item["dtype"]
            cpp_type = self._numpy_dtype_to_cpp(dtype)
            if cpp_type is None:
                raise ValueError(f"Unsupported NumPy dtype: {dtype}")
            if shape == ():
                type_str = cpp_type
            else:
                type_str = cpp_type + ''.join([f'[{x}]' for x in shape])

            lines.append(f"    template <>")
            lines.append(f"    struct field_info<{name}_tag> {{")
            lines.append(f"        using type = {type_str};")
            lines.append(f"        static constexpr std::size_t offset = {offset};")
            lines.append("    };")
            lines.append("")

        lines.append("} // namespace SharedMemoryLayout")
        lines.append("")
        lines.append("")

        lines.append("#define MAP_ALL_SHARED_MEMORY_FIELDS \\")
        #lines.append("      SharedMemoryAccess::Field{ \\")
        for item in self.layout_info:
            name = item["name"]
            offset = item["offset"]
            shape = item["shape"]
            dtype = item["dtype"]

            lines.append(f"     auto& {name} = get<SharedMemoryLayout::{name}_tag>();  \\")
        lines.append("")
        lines.append("")

        # Write to file
        code = "\n".join(lines)
        Path(output_file).write_text(code, encoding="utf-8")
        print(f"[SharedMemoryAllocator] Generated C++ header: {output_file}")


#%%
if __name__ == "__main__":
    allocator = SharedMemoryAllocator("shm_layout_example.json", create_new=True)
    #%%
    print("Fields in new shared memory:", list(allocator.fields.keys()))
    print("myint =", allocator.fields["myint"])
    print("myarr =\n", allocator.fields["myarr"])
    print("myarr2 =\n", allocator.fields["myarr2"])
    # Do not close/unlink yet if we want another process to attach...
    # allocator_new.close()
    # %%
    allocator.generate_cpp_header("src/shared_memory_layout.hxx")
    # %%
    allocator.close()
    # %%
