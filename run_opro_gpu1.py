#!/usr/bin/env python3
"""Run OPRO on GPU 1.

This script runs OPRO optimization on GPU 1 with predefined parameters.
It uses subprocess for proper process isolation and error handling.

Requires: GPU 1 to be available (CUDA_VISIBLE_DEVICES=1)
"""
import os
import subprocess
import sys


def main():
    # Set environment for GPU 1
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    cmd = [
        'uv', 'run', 'python', 'run_opro.py',
        '--model', 'qwen',
        '--budget', '200000',
        '--minibatch-size', '261',
        '--num-candidates', '8',
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"GPU: CUDA_VISIBLE_DEVICES=1")
    print("=" * 60)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
        )
        sys.exit(result.returncode)
    except FileNotFoundError as e:
        print(f"ERROR: Command not found: {e}")
        print("Make sure 'uv' is installed and run_opro.py exists.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
