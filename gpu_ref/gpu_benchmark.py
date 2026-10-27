#!/usr/bin/env python3
"""ResNet-50 FP32 batch-size sweep on the GPU (Layer 3).

For each batch size it measures, with CUDA events and a wall clock:
  * H2D   - host-to-device copy of the input batch
  * comp  - model forward pass (kernel-only time)
  * D2H   - device-to-host copy of the logits
  * e2e   - wall-clock time for the whole iteration (copy in, compute, copy out)
and, while the timed loop runs, samples board power with nvidia-smi.

Inputs are a fixed random tensor (timing does not depend on pixel values) and the
model uses random weights, so no dataset or download is needed. Accuracy is
measured separately by fp32_reference.py.

Each iteration is synchronized before the next starts, so the numbers describe a
single request at a time (latency-bound). That is stated in the output file.
"""
import argparse
import csv
import json
import statistics as st
import subprocess
import threading
import time
from pathlib import Path

import torch
from torchvision import models


class PowerSampler:
    """Background nvidia-smi sampler (watts), one reading per interval."""

    def __init__(self, gpu=0, interval_ms=100):
        self.cmd = ["nvidia-smi", "-i", str(gpu), "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits", "-lms", str(interval_ms)]
        self.samples, self._proc, self._thread = [], None, None

    def _read(self):
        for line in self._proc.stdout:
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                pass

    def start(self):
        self.samples = []
        self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def stop(self):
        self._proc.terminate()
        self._proc.wait()
        self._thread.join(timeout=2)
        return self.samples


def run_config(model, batch, seconds, warmup, pinned, device):
    x_host = torch.randn(batch, 3, 224, 224)
    if pinned:
        x_host = x_host.pin_memory()
    out_host = torch.empty(batch, 1000)
    if pinned:
        out_host = out_host.pin_memory()

    def one_iter(timed):
        e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        t0 = time.perf_counter()
        e[0].record()
        x = x_host.to(device, non_blocking=pinned)
        e[1].record()
        y = model(x)
        e[2].record()
        out_host.copy_(y, non_blocking=pinned)
        e[3].record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if not timed:
            return None
        return (e[0].elapsed_time(e[1]), e[1].elapsed_time(e[2]),
                e[2].elapsed_time(e[3]), (t1 - t0) * 1000.0)

    with torch.no_grad():
        for _ in range(warmup):
            one_iter(False)
        sampler = PowerSampler()
        sampler.start()
        time.sleep(0.5)                      # let the first power samples arrive
        rows, t_start = [], time.perf_counter()
        while time.perf_counter() - t_start < seconds:
            rows.append(one_iter(True))
        wall = time.perf_counter() - t_start
        watts = sampler.stop()

    h2d, comp, d2h, e2e = zip(*rows)
    watts = watts[2:] if len(watts) > 4 else watts   # drop warm-up samples
    avg_w = st.mean(watts) if watts else float("nan")
    ips = batch * len(rows) / wall
    return {
        "batch": batch, "iters": len(rows),
        "h2d_ms": st.mean(h2d), "compute_ms": st.mean(comp), "d2h_ms": st.mean(d2h),
        "e2e_ms_mean": st.mean(e2e), "e2e_ms_p50": st.median(e2e),
        "e2e_ms_p99": sorted(e2e)[int(0.99 * (len(e2e) - 1))],
        "images_per_s": ips, "avg_power_w": avg_w, "power_samples": len(watts),
        "images_per_s_per_w": ips / avg_w if watts else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seconds", type=float, default=15.0, help="timed seconds per run (default 15)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3, help="independent runs per batch size")
    ap.add_argument("--pageable", action="store_true", help="use pageable host memory instead of pinned")
    ap.add_argument("--out", type=Path, default=Path("bench_results"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True    # fine for timing; fixed shapes per run
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = models.resnet50(weights=None).eval().to(device).float()
    args.out.mkdir(parents=True, exist_ok=True)

    # idle baseline power
    s = PowerSampler()
    s.start()
    time.sleep(5)
    idle = s.stop()
    idle_w = st.mean(idle[2:]) if len(idle) > 4 else float("nan")
    print(f"Idle power: {idle_w:.1f} W")

    results = []
    for b in args.batches:
        for r in range(args.repeats):
            res = run_config(model, b, args.seconds, args.warmup, not args.pageable, device)
            res["repeat"] = r
            results.append(res)
            print(f"batch {b:>2} run {r}: {res['images_per_s']:8.1f} img/s  "
                  f"e2e {res['e2e_ms_mean']:7.2f} ms  comp {res['compute_ms']:7.2f} ms  "
                  f"h2d {res['h2d_ms']:6.2f} ms  {res['avg_power_w']:6.1f} W")

    cols = list(results[0].keys())
    with open(args.out / "gpu_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    meta = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "precision": "FP32",
        "host_memory": "pageable" if args.pageable else "pinned",
        "seconds_per_run": args.seconds, "repeats": args.repeats, "warmup_iters": args.warmup,
        "idle_power_w": idle_w, "power_source": "nvidia-smi power.draw, 100 ms interval",
        "note": "each iteration synchronized; latency-bound single-request timing; random weights and inputs",
    }
    (args.out / "gpu_benchmark_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {args.out}/gpu_benchmark.csv and gpu_benchmark_meta.json")


if __name__ == "__main__":
    main()
