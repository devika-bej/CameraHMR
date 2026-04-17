import numpy as np
import argparse

IMG_NAMES = ["front.jpg", "left.jpg", "right.jpg"]

def extract_and_stack(file_paths: list[str], i: int, output_path: str):
    archives = [np.load(f, allow_pickle=True) for f in file_paths]

    keys = set(archives[0].keys())
    for f, archive in zip(file_paths[1:], archives[1:]):
        if set(archive.keys()) != keys:
            raise ValueError(f"Keys mismatch between {file_paths[0]} and {f}")

    result = {}
    for key in keys:
        if key == "imgname":
            result[key] = np.array(IMG_NAMES)
            continue
        
        if key == "gt_keypoints":
            result[key] = []
            continue

        arrays = []
        for f, archive in zip(file_paths, archives):
            arr = archive[key]
            if i >= len(arr):
                raise IndexError(f"Index {i} out of bounds for key '{key}' in '{f}' (length {len(arr)})")
            arrays.append(arr[i])

        result[key] = np.stack(arrays, axis=0)

    np.savez(output_path, **result)
    print(f"Saved to {output_path}")
    for key, val in result.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the i-th row from each of 3 .npz files and stack them."
    )
    parser.add_argument("i", type=int, help="Row index to extract from each file")
    parser.add_argument("files", nargs=3, help="Paths to the 3 input .npz files")
    parser.add_argument(
        "--output", "-o", default="output.npz", help="Output .npz file path (default: output.npz)"
    )
    args = parser.parse_args()

    extract_and_stack(args.files, args.i, args.output)