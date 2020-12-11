"""Hash with SHA256 for PyTorch checkpoints."""

import hashlib
import os
import shutil
from argparse import ArgumentParser


def _hash_ckpt(ckpt_file: str, jitted: bool = False, output_path: str = ""):
    with open(ckpt_file, "rb") as file:
        sha_hash = hashlib.sha256(file.read()).hexdigest()

    path_to_file_list = ckpt_file.split(os.sep)
    ckpt_file_name = os.path.splitext(path_to_file_list[-1])[0]
    if jitted:
        filename = "-".join((ckpt_file_name, sha_hash[:8])) + ".ptc"
    else:
        filename = "-".join((ckpt_file_name, sha_hash[:8])) + ".pt"

    shutil.move(ckpt_file, os.path.join(output_path, filename))
    print(f"==> Saved state dict into {filename} | SHA256: {sha_hash}")

    return filename, sha_hash


if __name__ == "__main__":
    parser = ArgumentParser(description="Hash checkpointed PyTorch file.")
    parser.add_argument(
        "--ckpt_file",
        type=str,
        required=True,
        help="Path to the checkpoint file including filename",
    )
    parser.add_argument(
        "--jitted",
        type=bool,
        required=False,
        default=False,
        help="Is the checkpoint file jitted? (.ptc for jitted else .pt)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=os.getcwd(),
        help="Path to the hashed file",
    )
    args = parser.parse_args()
    _hash_ckpt(args.ckpt_file, args.jitted, args.output_path)
