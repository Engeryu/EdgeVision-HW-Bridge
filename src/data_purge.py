# ===========================================================
#  File    : data_purge.py
#  Author  : engeryu
#  Created : 2026-03-15
# ===========================================================
#  Releases GPU VRAM and system RAM held by PyTorch.
#  Useful after training or co-simulation to free resources
#  without restarting the Python process.
#
#  Usage:
#    uv run python purge.py
# ===========================================================

import ctypes
import gc

import torch


def purge_memory() -> None:
    """
    Releases all GPU VRAM and system RAM held by PyTorch.

    Performs the following steps in order:
    1. Runs Python's garbage collector to dereference unused tensors.
    2. Empties the PyTorch CUDA memory cache (NVIDIA / ROCm).
    3. Resets accumulated memory statistics.
    4. Attempts to release fragmented system RAM back to the OS
       via malloc_trim (Linux only, no-op on other platforms).
    """
    # ── Step 1: Python garbage collection ─────────────────
    collected = gc.collect()
    print(f"[GC]   Collected {collected} unreachable objects.")

    # ── Step 2: CUDA / ROCm VRAM release ──────────────────
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        before = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated() / 1024**2
        print(f"[CUDA] Device     : {device_name}")
        print(f"[CUDA] Allocated  : {before:.1f} MB → {after:.1f} MB")
        print(f"[CUDA] Cache freed: {before - after:.1f} MB")

        # Reset internal memory stats (peak, total, etc.)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        print("[CUDA] Memory statistics reset.")
    else:
        print("[CUDA] No CUDA device detected — skipping VRAM purge.")

    # ── Step 3: MPS (Apple Silicon) release ───────────────
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("[MPS]  MPS cache cleared.")

    # ── Step 4: System RAM release (Linux only) ───────────
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        print("[RAM]  malloc_trim called — fragmented RAM returned to OS.")
    except OSError:
        print("[RAM]  malloc_trim not available (non-Linux platform) — skipped.")

    print("\n✔  Purge complete.")


def main():
    purge_memory()


if __name__ == "__main__":
    main()
