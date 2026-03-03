#!/usr/bin/env python3
"""
Lightweight wrapper to run the existing training entrypoint while sampling memory (CPU+GPU)
and writing a CSV memory log into the training stats directory (or cwd if not set).

Usage: python train_with_profiler.py --config path/to/config.yaml --model MODEL_NAME [--device cuda]

This script reuses the `train.main()` entrypoint from `train.py` and starts a background
MemProfiler that writes a CSV file named memory_profile_<timestamp>.csv.
"""
import os
from datetime import datetime
from train import parse_args, main as train_main
from utils import file_utils
from utils.mem_profiler import MemProfiler


def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    # Determine stats dir from config (same key used by train.py)
    stats_dir = None
    if 'training_parameters' in config:
        stats_dir = config['training_parameters'].get('stats_dir', None)

    curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mem_log_dir = stats_dir if stats_dir is not None else os.getcwd()
    if not os.path.exists(mem_log_dir):
        os.makedirs(mem_log_dir, exist_ok=True)

    mem_log_path = os.path.join(mem_log_dir, f'memory_profile_{curr_date_str}.csv')
    profiler = MemProfiler(mem_log_path, interval=1.0)

    try:
        profiler.start()
        # Call the existing training main() which will parse args again and run the full flow.
        train_main()
    finally:
        profiler.stop()


if __name__ == '__main__':
    main()
