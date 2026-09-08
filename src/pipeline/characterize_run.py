#!/usr/bin/env python3
"""
characterize_run.py — Extract 1 Hz video + audio features for one or more runs.

Per-second features saved as .npz in $CS_CACHE/cache_1hz/:
  VIDEO  luminance         mean grayscale [0-255]
         frame_diff        mean |Δframe| (motion proxy)
  AUDIO  rms               RMS energy
         spectral_centroid  brightness (Hz)
         spectral_flatness  Wiener entropy (high=noise/speech, low=tonal/music)
         zcr               zero-crossing rate
         hf_ratio          energy above 4 kHz / total (high = sibilants → speech)

Usage:
    python src/characterize_run.py --run-id friends_s06e01a   # one run
    python src/characterize_run.py --task-id 0                # SLURM array
    python src/characterize_run.py --all                      # sequential
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
#  Video
# ---------------------------------------------------------------------------

def video_features_streaming(video_path: str):
    """Decode at 1 fps and compute features on the fly (< 5 MB RAM)."""
    import imageio_ffmpeg

    gen = imageio_ffmpeg.read_frames(
        video_path,
        pix_fmt="rgb24",
        output_params=["-vf", "fps=1"],
    )
    meta = next(gen)
    w, h = meta["size"]
    _W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    luminance_list = []
    frame_diff_list = []
    prev_gray = None

    for raw in gen:
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        gray = np.dot(rgb, _W)
        luminance_list.append(gray.mean())
        if prev_gray is not None:
            frame_diff_list.append(np.abs(gray - prev_gray).mean())
        else:
            frame_diff_list.append(0.0)
        prev_gray = gray

    vf = {
        "luminance": np.array(luminance_list, dtype=np.float32),
        "frame_diff": np.array(frame_diff_list, dtype=np.float32),
    }
    return vf, meta
# ---------------------------------------------------------------------------
#  Audio
# ---------------------------------------------------------------------------

def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Read WAV → float32 mono. Uses scipy (no librosa/soundfile needed)."""
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def audio_features(audio: np.ndarray, sr: int, max_t: int) -> dict[str, np.ndarray]:
    """Compute per-second audio features. max_t caps output length."""
    win = sr                               # 1-second window
    T = min(len(audio) // win, max_t)

    rms              = np.zeros(T, dtype=np.float32)
    spectral_centroid = np.zeros(T, dtype=np.float32)
    spectral_flatness = np.zeros(T, dtype=np.float32)
    zcr              = np.zeros(T, dtype=np.float32)
    hf_ratio         = np.zeros(T, dtype=np.float32)

    hf_bin = int(4000 * win / sr)          # bin index for 4 kHz

    for i in range(T):
        chunk = audio[i * win : (i + 1) * win]

        # RMS
        rms[i] = np.sqrt(np.mean(chunk ** 2))

        # ZCR
        signs = np.sign(chunk)
        zcr[i] = np.mean(np.abs(np.diff(signs)) > 0)

        # Spectrum
        spec = np.abs(np.fft.rfft(chunk))
        spec = spec[1:]                    # drop DC
        spec_sum = spec.sum() + 1e-10
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)[1:]

        # Centroid (Hz)
        spectral_centroid[i] = (freqs * spec).sum() / spec_sum

        # Flatness (Wiener entropy)
        log_spec = np.log(spec + 1e-10)
        spectral_flatness[i] = np.exp(log_spec.mean()) / (spec.mean() + 1e-10)

        # HF energy ratio (>4 kHz)
        if hf_bin < len(spec):
            hf_ratio[i] = spec[hf_bin:].sum() / spec_sum

    return {
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "spectral_flatness": spectral_flatness,
        "zcr": zcr,
        "hf_ratio": hf_ratio,
    }


def nan_audio(T: int) -> dict[str, np.ndarray]:
    """Placeholder when .wav is missing."""
    return {k: np.full(T, np.nan, dtype=np.float32)
            for k in ("rms", "spectral_centroid", "spectral_flatness", "zcr", "hf_ratio")}


# ---------------------------------------------------------------------------
#  Per-run pipeline
# ---------------------------------------------------------------------------

def process_run(run: dict, cache_dir: Path, force: bool = False) -> Path:
    run_id = run["run_id"]
    out_path = cache_dir / f"{run_id}.npz"

    if out_path.exists() and not force:
        print(f"  {run_id}: cached → skip")
        return out_path

    t0 = time.time()

    # --- video ---
    print(f"  {run_id}: decode video @ 1 Hz ...", flush=True)
    vf, vmeta = video_features_streaming(run["video"])
    T_vid = len(vf["luminance"])

    # --- audio ---
    if run["audio"]:
        print(f"  {run_id}: audio features ...", flush=True)
        audio, sr = read_wav_mono(run["audio"])
        af = audio_features(audio, sr, max_t=T_vid)
        T_aud = len(af["rms"])
        del audio
    else:
        print(f"  {run_id}: ⚠ no .wav → NaN")
        af = nan_audio(T_vid)
        T_aud = T_vid

    # --- align & save ---
    T = min(T_vid, T_aud)

    np.savez_compressed(
        out_path,
        # timestamps
        time_s=np.arange(T, dtype=np.float32),
        # video
        luminance=vf["luminance"][:T],
        frame_diff=vf["frame_diff"][:T],
        # audio
        rms=af["rms"][:T],
        spectral_centroid=af["spectral_centroid"][:T],
        spectral_flatness=af["spectral_flatness"][:T],
        zcr=af["zcr"][:T],
        hf_ratio=af["hf_ratio"][:T],
        # metadata (stored as 0-d arrays by savez)
        run_id=run_id,
        film_id=run["film_id"],
        video_w=vmeta["size"][0],
        video_h=vmeta["size"][1],
    )

    dt = time.time() - t0
    print(f"  {run_id}: done — T={T}s, {dt:.1f}s elapsed")
    return out_path


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def load_run_index() -> list[dict]:
    code = os.environ["CS_CODE"]
    with open(Path(code) / "manifest" / "run_index.json") as f:
        return json.load(f)["runs"]


def main():
    p = argparse.ArgumentParser(description="1 Hz feature extraction per run")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="Single run by ID")
    g.add_argument("--task-id", type=int, help="SLURM_ARRAY_TASK_ID")
    g.add_argument("--all", action="store_true", help="All runs, sequential")
    p.add_argument("--force", action="store_true", help="Overwrite cached .npz")
    args = p.parse_args()

    runs = load_run_index()
    cache_dir = Path(os.environ["CS_CACHE"]) / "cache_1hz"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = runs
    elif args.run_id:
        targets = [r for r in runs if r["run_id"] == args.run_id]
        if not targets:
            sys.exit(f"Run '{args.run_id}' not in index")
    else:
        if args.task_id >= len(runs):
            sys.exit(f"task-id {args.task_id} out of range (0..{len(runs)-1})")
        targets = [runs[args.task_id]]

    print(f"Processing {len(targets)} run(s) → {cache_dir}")
    for run in targets:
        process_run(run, cache_dir, force=args.force)
    print("All done.")


if __name__ == "__main__":
    main()
