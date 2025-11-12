"""
Download LIBERO datasets from Hugging Face.
Run this from the project root directory.
"""
from huggingface_hub import snapshot_download
import os
import sys

# Change to parent directory to avoid import conflicts with data/filelock.py
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def download_libero_datasets():
    """Download libero_10 and libero_90 datasets."""
    repo_id = "yifengzhu-hf/LIBERO-datasets"
    save_dir = "data/datasets"
    datasets = ["libero_10", "libero_90"]

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Downloading {dataset_name}...")
        print(f"{'='*60}")

        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=f"{dataset_name}/*",
                local_dir=save_dir,
                local_dir_use_symlinks=False,
            )
            print(
                f"✓ Successfully downloaded {dataset_name} to {os.path.join(save_dir, dataset_name)}")
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Download completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    download_libero_datasets()
