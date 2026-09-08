#!/usr/bin/env python3
"""
extract_latents.py — Slice fMRI clips WITH THE BOLD LAG and run CortexMAE-F.

=====================  THE LAG — READ THIS  =====================
Both the stimulus features and the flattened fMRI live on the same 1 Hz
timeline, index 0 = run onset.

The BOLD response to the stimulus at second s appears in the fMRI at s + LAG.
Therefore, for a stimulus clip [start_s, start_s + 16):

    fMRI window = [start_s + LAG,  start_s + LAG + 16)     <-- ADD the lag

We shift the fMRI window FORWARD. Subtracting would grab fMRI that PRECEDES
the stimulus — the classic sign error. Run validate_lag.py first to confirm
LAG empirically on the data.
================================================================

Latents are saved UNPOOLED — patch_embeds [4, ~364, 768] — so spatial pooling,
Yeo-7 pooling and flattening can all be explored later without re-running the GPU.

Usage:
    python src/extract_latents.py --lag 5
    python src/extract_latents.py --lag 5 --task-id 0    # SLURM array over subjects
"""

import argparse, json, os, sys, time
import numpy as np
import torch
from pathlib import Path

NUM_FRAMES = 16          # CortexMAE input is always 16 s
FLAT_DIM   = 77763


def load_manifest(code: Path):
    with open(code / "manifest" / "clip_manifest.json") as f:
        return json.load(f)["clips"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, required=True,
                   help="BOLD lag in seconds (fMRI window = stimulus window + lag)")
    p.add_argument("--subject", help="e.g. sub-01 (default: all)")
    p.add_argument("--task-id", type=int, help="SLURM array index over subjects")
    p.add_argument("--model", default="cortex_mae_flat")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    code  = Path(os.environ["CS_CODE"])
    cache = Path(os.environ["CS_CACHE"])
    fdir  = cache / "fmri_flat_1hz"
    out_dir = cache / f"latents_lag{args.lag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_manifest(code)
    subjects = sorted({p.stem.split("_", 1)[0] for p in fdir.glob("*.npy")})
    if not subjects:
        sys.exit(f"No flattened fMRI in {fdir} — run flatten_fmri.py first")

    if args.task_id is not None:
        if args.task_id >= len(subjects):
            sys.exit(f"task-id {args.task_id} out of range (0..{len(subjects)-1})")
        subjects = [subjects[args.task_id]]
    elif args.subject:
        subjects = [args.subject]

    print(f"LAG = {args.lag}s   (fMRI window = [start_s + {args.lag}, start_s + {args.lag + NUM_FRAMES}))")
    print(f"{len(clips)} clips × {len(subjects)} subject(s) → {out_dir}\n")

    from cortex_mae import CortexMAE
    model = CortexMAE.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.set_device(device)
    print(f"model={args.model}  device={device}\n")

    n_ok = n_skip = n_oob = 0

    for sub in subjects:
        run_cache: dict[str, np.ndarray] = {}

        for clip in clips:
            run_id  = clip["run_id"]
            clip_id = clip["clip_id"]
            out = out_dir / f"{sub}__{clip_id}.npz"
            if out.exists() and not args.force:
                n_skip += 1
                continue

            # --- load the run's flat 1 Hz fMRI (cached in RAM per run) ---
            if run_id not in run_cache:
                npy = fdir / f"{sub}_{run_id}.npy"
                if not npy.exists():
                    run_cache[run_id] = None
                else:
                    run_cache[run_id] = np.load(npy).astype(np.float32)
            fmri = run_cache[run_id]
            if fmri is None:
                continue                       # this subject lacks this run

            # ============ THE LAG: fMRI window = stimulus window + LAG ============
            t0 = clip["start_s"] + args.lag
            t1 = t0 + NUM_FRAMES
            # =====================================================================

            T = fmri.shape[0]
            if t0 < 0 or t1 > T:
                n_oob += 1
                continue                       # window runs off the end of the run

            bold = fmri[t0:t1]                 # (16, 77763)
            assert bold.shape == (NUM_FRAMES, FLAT_DIM), bold.shape

            # dict input bypasses read_sample(), so we z-score the window ourselves
            # (the run was already z-scored + detrended; this is the clip-level scale
            #  CortexMAE's own read_sample would have applied)
            mu = bold.mean(axis=0, keepdims=True)
            sd = bold.std(axis=0, keepdims=True)
            bold = np.where(sd > 1e-6, (bold - mu) / np.clip(sd, 1e-6, None), 0.0)

            # emulate what read_sample() would produce: bold + mean/std used by Transform
            sample = {
                "bold": torch.from_numpy(bold.astype(np.float16)),
                "mean": torch.from_numpy(mu.astype(np.float32).squeeze(0)),
                "std":  torch.from_numpy(sd.astype(np.float32).squeeze(0)),
                "tr":   1.0,
            }

            embeds = model.run_embedding(sample, tr=1.0, batch_size=args.batch_size)
            patch = embeds.patch_embeds.float().numpy()      # (clips, tokens, 768)
            patch = patch[0] if patch.shape[0] == 1 else patch

            np.savez_compressed(
                out,
                patch_embeds=patch.astype(np.float32),       # UNPOOLED
                cls_embeds=(embeds.cls_embeds.float().numpy()
                            if embeds.cls_embeds is not None else np.array([])),
                subject=sub,
                clip_id=clip_id,
                run_id=run_id,
                film_id=clip["film_id"],
                content_label=clip["content_label"],
                start_s=clip["start_s"],
                lag=args.lag,
                fmri_window=np.array([t0, t1]),
            )
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  {n_ok} done  (last: {sub}/{clip_id}  patch={patch.shape})", flush=True)

        run_cache.clear()

    print(f"\nextracted={n_ok}  skipped(cached)={n_skip}  out-of-bounds={n_oob}")
    print(f"→ {out_dir}")
    if n_oob:
        print(f"  note: {n_oob} clip×subject windows fell off the end of their run "
              f"(start_s + {args.lag} + 16 > run length)")


if __name__ == "__main__":
    main()
