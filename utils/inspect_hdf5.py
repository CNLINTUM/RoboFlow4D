# inspect_single_h5.py
import h5py
import sys
from pathlib import Path

def print_h5_structure(h5_path: str):
    h5_path = Path(h5_path)
    print(f"\n=== Inspecting: {h5_path} ===")

    def _print(name, obj):
        prefix = "  "
        if isinstance(obj, h5py.Group):
            print(f"{prefix}[Group ] /{name}")
            for k, v in obj.attrs.items():
                print(f"{prefix}  @attr {k} = {v!r}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}[Dataset] /{name} shape={obj.shape} dtype={obj.dtype}")
            for k, v in obj.attrs.items():
                print(f"{prefix}  @attr {k} = {v!r}")

    with h5py.File(h5_path, "r") as f:
        # root attrs
        print("[Root attrs]")
        for k, v in f.attrs.items():
            print(f"  @attr {k} = {v!r}")
        print("\n[Tree]")
        f.visititems(_print)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_single_h5.py <file.hdf5>")
        sys.exit(1)
    print_h5_structure(sys.argv[1])