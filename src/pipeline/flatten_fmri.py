#!/usr/bin/env python3
"""
flatten_fmri.py — Volume MNI → fsLR surface → CortexMAE flat map → 1 Hz cache.

This is the EXPENSIVE stage. It runs once per (subject, run) and caches the
result, so the BOLD lag stays a free parameter downstream.

Pipeline (order matters — z-score/detrend BEFORE resampling):
  1. load 4D .nii.gz (MNI152NLin2009cAsym, TR = 1.49 s)
  2. neuromaps mni152_to_fslr → surface (T, 64984)
  3. CortexMAE FlatResampler → flat map → mask → (T, 77763)
  4. z-score per pixel over the full run
  5. linear detrend
  6. resample to 1 Hz (linear interp)
  7. save float16 → $CS_CACHE/fmri_flat_1hz/{subject}_{run_id}.npy

Output is on the SAME 1 Hz timeline as the stimulus features, with index 0 =
run onset. NO lag is applied here.

Usage:
    python src/flatten_fmri.py --task-id 0     # SLURM array
    python src/flatten_fmri.py --index 0       # single, for testing
"""

import argparse, json, os, sys, time
import numpy as np
from pathlib import Path

TR_NATIVE = 1.49


def load_entries() -> list[dict]:
    code = Path(os.environ["CS_CODE"])
    with open(code / "manifest" / "fmri_index.json") as f:
        return json.load(f)["entries"]


def project_to_flat(nii_path: str, reader) -> np.ndarray:
    """Volume MNI → fsLR surface → flat map. Returns (T, 77763) float32."""
    import nibabel as nib
    from neuromaps.transforms import mni152_to_fslr

    img = nib.load(nii_path)
    print(f"      volume shape {img.shape}", flush=True)

    t0 = time.time()
    lh, rh = mni152_to_fslr(img, fslr_density="32k", method="linear")
    surf = np.stack([
        np.concatenate([dl.data, dr.data])
        for dl, dr in zip(lh.darrays, rh.darrays)
    ]).astype(np.float32)                          # (T, 64984)
    print(f"      → surface {surf.shape}  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    flat = reader.resampler.transform(surf, interpolation="linear")   # (T, 224, 560)
    flat = flat[:, reader.resampler.mask_].astype(np.float32)         # (T, 77763)
    print(f"      → flat {flat.shape}  [{time.time()-t0:.0f}s]", flush=True)
    return flat


def normalize_and_resample(flat: np.ndarray) -> np.ndarray:
    """z-score → detrend → resample to 1 Hz. Input (T, D) at TR_NATIVE."""
    from scipy.signal import detrend as sp_detrend
    from scipy.interpolate import interp1d

    # 1. z-score per pixel over the run (guard against zero-variance pixels)
    mean = flat.mean(axis=0, keepdims=True)
    std  = flat.std(axis=0, keepdims=True)
    valid = std > 1e-6
    flat = np.where(valid, (flat - mean) / np.clip(std, 1e-6, None), 0.0)

    # 2. linear detrend
    flat = sp_detrend(flat, axis=0, type="linear")

    # 3. resample TR_NATIVE → 1 Hz
    T = flat.shape[0]
    t_native = np.arange(T) * TR_NATIVE
    t_new    = np.arange(0, int(np.floor(t_native[-1])) + 1, 1.0)
    f = interp1d(t_native, flat, axis=0, kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    return f(t_new).astype(np.float32)


def process(entry: dict, reader, out_dir: Path, force: bool = False):
    sub, run_id = entry["subject"], entry["run_id"]
    out = out_dir / f"{sub}_{run_id}.npy"
    if out.exists() and not force:
        print(f"  {sub}/{run_id}: cached → skip")
        return

    print(f"  {sub}/{run_id}:", flush=True)
    t0 = time.time()

    flat = project_to_flat(entry["path"], reader)
    T_native = flat.shape[0]
    flat = normalize_and_resample(flat)
    T_1hz = flat.shape[0]

    np.save(out, flat.astype(np.float16))
    print(f"      T {T_native} @{TR_NATIVE}s → {T_1hz} @1Hz  "
          f"({T_1hz}s)  [{time.time()-t0:.0f}s total]  → {out.name}\n", flush=True)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task-id", type=int, help="SLURM_ARRAY_TASK_ID")
    g.add_argument("--index",   type=int, help="single entry index")
    g.add_argument("--all",     action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    entries = load_entries()
    out_dir = Path(os.environ["CS_CACHE"]) / "fmri_flat_1hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = entries
    else:
        i = args.task_id if args.task_id is not None else args.index
        if i >= len(entries):
            sys.exit(f"index {i} out of range (0..{len(entries)-1})")
        targets = [entries[i]]

    from cortex_mae.inference import get_reader
    reader = get_reader("flat")
    print(f"FlatResampler: {reader.resampler.mask_.sum()} valid pixels\n")

    print(f"Processing {len(targets)} (subject, run) pair(s) → {out_dir}\n")
    for e in targets:
        process(e, reader, out_dir, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
