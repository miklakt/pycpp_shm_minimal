from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shm_allocator import SharedMemoryAllocator


def main():
    script_dir = Path(__file__).resolve().parent
    spec_file = script_dir / "shm_layout_example.json"
    header_file = script_dir / "shared_memory_layout.hxx"

    allocator = SharedMemoryAllocator(spec_file, create_new=True)
    allocator.generate_cpp_header(header_file)

    print("Fields in shared memory:", list(allocator.fields.keys()))
    print("Header generated:", header_file)
    print("Shared memory is ready.")
    print("Keep this script running while you execute the C++ binary.")
    print("Example binary command from project root:")
    print("  ./cpp_examples/bin/access_by_name_example")
    print("")
    print("Press Enter to unlink shared memory and exit.")

    try:
        input()
    finally:
        allocator.close(unlink=True)


if __name__ == "__main__":
    main()
