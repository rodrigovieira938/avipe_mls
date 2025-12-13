import os
from pathlib import Path
import tarfile


def run(dataset_path: str):
    if os.path.exists(f"{dataset_path}/data/grapevista"):
        return
    parts = sorted(Path(f"{dataset_path}/data/").glob("grapevista.tar.*.gz.part"))  # Sort to ensure correct order
    print("Concatting parts")
    with open(f"{dataset_path}/data/grapevista.tar.gz", 'wb') as f_out:
        for part in parts:
            with open(part, 'rb') as f_in:
                while chunk := f_in.read(1024 * 1024):
                    f_out.write(chunk)
    print("Decompressing")
    with tarfile.open(f"{dataset_path}/data/grapevista.tar.gz", 'r:gz') as tar:
        tar.extractall(path=f"{dataset_path}/data")
    print("Decompression complete")