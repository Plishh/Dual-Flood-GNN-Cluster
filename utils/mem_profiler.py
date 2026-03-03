import threading
import time
import csv
import subprocess
import os
from datetime import datetime

try:
    import psutil
except Exception:
    psutil = None

import torch


class MemProfiler:
    """Background sampler that writes periodic memory stats (CPU RSS, CUDA, nvidia-smi) to a CSV file.

    Usage:
        profiler = MemProfiler(output_path, interval=1.0)
        profiler.start()
        ... training ...
        profiler.stop()
    """

    def __init__(self, output_path: str, interval: float = 1.0):
        self.output_path = output_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # ensure directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # write header
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_utc',
                'elapsed_s',
                'rss_mb',
                'cuda_available',
                'cuda_allocated_mb',
                'cuda_reserved_mb',
                'nvidia_mem_used_mb',
                'nvidia_mem_total_mb',
                'gpu_util_percent',
            ])

        self._start_time = None

    def start(self):
        self._start_time = time.time()
        self._stop_event.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _sample_nvidia_smi(self):
        """Return tuple (used_mb, total_mb, util_percent) or (None, None, None) if nvidia-smi missing."""
        try:
            out = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], stderr=subprocess.DEVNULL)
            # may report multiple GPUs; take GPU 0 first line
            line = out.decode('utf-8').strip().splitlines()[0]
            used, total, util = [float(x.strip()) for x in line.split(',')]
            return used, total, util
        except Exception:
            return None, None, None

    def _run(self):
        while not self._stop_event.is_set():
            ts = datetime.utcnow().isoformat()
            elapsed = None if self._start_time is None else (time.time() - self._start_time)

            rss_mb = None
            if psutil is not None:
                try:
                    rss_mb = psutil.Process().memory_info().rss / (1024**2)
                except Exception:
                    rss_mb = None

            cuda_available = torch.cuda.is_available()
            cuda_alloc = None
            cuda_res = None
            if cuda_available:
                try:
                    cuda_alloc = torch.cuda.memory_allocated() / (1024**2)
                    cuda_res = torch.cuda.memory_reserved() / (1024**2)
                except Exception:
                    cuda_alloc = None
                    cuda_res = None

            n_used, n_total, n_util = self._sample_nvidia_smi()

            with open(self.output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts,
                    f'{elapsed:.3f}' if elapsed is not None else '',
                    f'{rss_mb:.3f}' if rss_mb is not None else '',
                    int(cuda_available),
                    f'{cuda_alloc:.3f}' if cuda_alloc is not None else '',
                    f'{cuda_res:.3f}' if cuda_res is not None else '',
                    f'{n_used:.3f}' if n_used is not None else '',
                    f'{n_total:.3f}' if n_total is not None else '',
                    f'{n_util:.1f}' if n_util is not None else '',
                ])

            time.sleep(self.interval)
