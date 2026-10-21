"""Command-line interface for causal attention (Task 4, owner: Harshith).

Usage:
    python run_attention.py --input qkv.npz --output out.npy

Reads Q, K, V from an .npz file, validates dtype, shape and tile divisibility
before launch (see attention/validation.py), runs the kernel and writes the
output as .npy. Unsupported inputs exit with a nonzero status and a clear message.
"""

import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run causal attention on Q, K, V from an .npz file.")
    parser.add_argument("--input", required=True, help="Path to .npz containing q, k, v")
    parser.add_argument("--output", required=True, help="Path for the output .npy")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # TODO(T4): load npz, call attention.validation.validate_qkv, run kernel, save .npy
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
