# import argparse
# import os
# import textwrap
# from typing import Iterable, Optional

# import h5py
# import numpy as np


# def iter_hdf5_files(path: str, max_files: Optional[int]) -> Iterable[str]:
#     """Yield HDF5 file paths under `path` (file or directory)."""
#     if os.path.isfile(path):
#         if path.lower().endswith((".hdf5", ".h5")):
#             yield os.path.abspath(path)
#         else:
#             raise ValueError(f"Provided path is not an HDF5 file: {path}")
#         return

#     if not os.path.isdir(path):
#         raise FileNotFoundError(f"Path does not exist: {path}")

#     count = 0
#     for entry in sorted(os.listdir(path)):
#         full_path = os.path.join(path, entry)
#         if os.path.isfile(full_path) and full_path.lower().endswith((".hdf5", ".h5")):
#             yield os.path.abspath(full_path)
#             count += 1
#             if max_files is not None and count >= max_files:
#                 break


# def _format_attr_value(value) -> str:
#     if isinstance(value, bytes):
#         value = value.decode("utf-8", errors="replace")

#     if isinstance(value, str):
#         if len(value) > 200:
#             return repr(value[:197] + "...")
#         return repr(value)

#     if isinstance(value, (np.generic,)):
#         return repr(value.item())

#     if isinstance(value, np.ndarray):
#         if value.ndim == 0:
#             return repr(value.item())
#         preview_elems = ", ".join(f"{x:.6g}" for x in value.flatten()[:6])
#         if value.size > 6:
#             preview_elems += ", ..."
#         return (
#             f"array(shape={value.shape}, dtype={value.dtype}, "
#             f"preview=[{preview_elems}])"
#         )

#     if isinstance(value, (list, tuple)):
#         preview_elems = ", ".join(repr(v) for v in value[:6])
#         if len(value) > 6:
#             preview_elems += ", ..."
#         return f"{type(value).__name__}([{preview_elems}])"

#     return repr(value)


# def describe_attrs(obj, indent: str) -> None:
#     """Print HDF5 attributes with indentation."""
#     if not obj.attrs:
#         return

#     print(f"{indent}Attributes:")
#     for key in sorted(obj.attrs.keys()):
#         value = obj.attrs[key]
#         formatted = _format_attr_value(value)
#         print(f"{indent}  - {key}: {formatted}")


# def describe_dataset(dataset: h5py.Dataset, indent: str, include_stats: bool) -> None:
#     """Print dataset summary information."""
#     shape = dataset.shape
#     dtype = dataset.dtype
#     print(f"{indent}Dataset: shape={shape}, dtype={dtype}")
#     describe_attrs(dataset, indent + "  ")

#     if include_stats and dataset.size > 0 and dataset.dtype.kind in {"i", "u", "f"}:
#         # Avoid loading huge datasets entirely by sampling the first axis.
#         sample = dataset[()]
#         stats_indent = indent + "  "
#         print(
#             f"{stats_indent}Stats: min={sample.min():.6g}, "
#             f"max={sample.max():.6g}, mean={sample.mean():.6g}"
#         )


# def describe_group(
#     group: h5py.Group,
#     indent: str,
#     depth: int,
#     max_items: Optional[int],
#     include_stats: bool,
# ) -> None:
#     """Recursively print group contents."""
#     describe_attrs(group, indent)

#     if depth == 0:
#         return

#     keys = sorted(group.keys())
#     if max_items is not None:
#         keys = keys[:max_items]

#     for key in keys:
#         obj = group[key]
#         print(f"{indent}{key}/")
#         if isinstance(obj, h5py.Group):
#             describe_group(
#                 obj,
#                 indent + "  ",
#                 depth=depth - 1 if depth > 0 else depth,
#                 max_items=max_items,
#                 include_stats=include_stats,
#             )
#         elif isinstance(obj, h5py.Dataset):
#             describe_dataset(obj, indent + "  ", include_stats)
#         else:
#             print(f"{indent}  (unknown HDF5 object type: {type(obj)})")


# def summarize_file(
#     file_path: str,
#     max_depth: int,
#     max_items: Optional[int],
#     max_demos: Optional[int],
#     include_stats: bool,
# ) -> None:
#     """Print a formatted summary of a single HDF5 file."""
#     print("=" * 80)
#     print(f"File: {file_path}")
#     print("=" * 80)

#     with h5py.File(file_path, "r") as f:
#         top_keys = sorted(f.keys())
#         print(
#             f"Top-level keys: {', '.join(top_keys) if top_keys else '(none)'}")
#         describe_attrs(f, indent="")

#         # For LIBERO-style files, demos are under data/demo_X/
#         if "data" in f and isinstance(f["data"], h5py.Group):
#             data_group = f["data"]
#             demo_keys = sorted(data_group.keys())
#             if max_demos is not None:
#                 demo_keys = demo_keys[:max_demos]

#             print("\n--- Demo groups ---")
#             print(
#                 textwrap.fill(
#                     "Listing up to "
#                     f"{len(demo_keys)} demo(s). Use --max-demos to control this.",
#                     width=78,
#                 )
#             )

#             for demo in demo_keys:
#                 demo_group = data_group[demo]
#                 print(f"\n[data/{demo}]")
#                 describe_group(
#                     demo_group,
#                     indent="  ",
#                     depth=max_depth,
#                     max_items=max_items,
#                     include_stats=include_stats,
#                 )
#         else:
#             print("\n-- File contents --")
#             describe_group(
#                 f,
#                 indent="  ",
#                 depth=max_depth,
#                 max_items=max_items,
#                 include_stats=include_stats,
#             )


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Inspect the structure of LIBERO HDF5 demo files."
#     )
#     parser.add_argument(
#         "--path",
#         default="data/datasets/libero_10",
#         help="Path to an HDF5 file or a directory containing .hdf5 files.",
#     )
#     parser.add_argument(
#         "--max-files",
#         type=int,
#         default=1,
#         help="Maximum number of files to inspect when `path` is a directory.",
#     )
#     parser.add_argument(
#         "--max-demos",
#         type=int,
#         default=1,
#         help="Maximum number of demo_* groups to display inside each file.",
#     )
#     parser.add_argument(
#         "--max-depth",
#         type=int,
#         default=2,
#         help="How deep to traverse within each demo group.",
#     )
#     parser.add_argument(
#         "--max-items",
#         type=int,
#         default=10,
#         help="Limit the number of items listed per group.",
#     )
#     parser.add_argument(
#         "--stats",
#         action="store_true",
#         help="Compute basic statistics (min/max/mean) for numeric datasets. "
#         "This may load entire datasets into memory.",
#     )
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     try:
#         files = list(iter_hdf5_files(args.path, args.max_files))
#     except (FileNotFoundError, ValueError) as exc:
#         print(f"Error: {exc}")
#         return

#     if not files:
#         print(f"No HDF5 files found under {args.path}")
#         return

#     for file_path in files:
#         summarize_file(
#             file_path=file_path,
#             max_depth=max(args.max_depth, 0),
#             max_items=args.max_items if args.max_items > 0 else None,
#             max_demos=args.max_demos if args.max_demos > 0 else None,
#             include_stats=args.stats,
#         )


# if __name__ == "__main__":
#     main()
import h5py

file_path = "data/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"

with h5py.File(file_path, "r") as f:
    if "data" in f:
        g = f["data"]
        print("=== data group 的 attrs ===")
        for k, v in g.attrs.items():
            print(f"{k}: {v}")
    else:
        print("这个文件里没有 'data' group")
