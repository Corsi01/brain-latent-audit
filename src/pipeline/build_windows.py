#!/usr/bin/env python3
"""
build_windows.py — Slide 16s windows over cached 1 Hz features + transcripts.

Reads:  $CS_CACHE/cache_1hz/*.npz       (video + audio features)
        $CS_STIMULI/../transcripts/      (word-level timestamps)
Writes: $CS_CODE/manifest/candidate_windows.json

Each window has clip-level (mean/std) and block-level (4×4s) features
including speech_coverage and word_rate from transcripts.

Usage:
    source env.sh && python src/build_windows.py
"""

import ast, json, os, sys
import numpy as np
from pathlib import Path

WINDOW_S = 16
BLOCK_S  = 4
STRIDE_S = 4
N_BLOCKS = WINDOW_S // BLOCK_S

AV_FEATURES = [
    "luminance", "frame_diff",
    "rms", "spectral_centroid", "spectral_flatness", "zcr", "hf_ratio",
]
TRANSCRIPT_FEATURES = ["speech_coverage", "word_rate"]
ALL_FEATURES = AV_FEATURES + TRANSCRIPT_FEATURES


# ------------------------------------------------------------------
#  Transcript parsing
# ------------------------------------------------------------------

def find_transcript(run_id: str, film_id: str, stimuli_root: str) -> str | None:
    """Locate the TSV transcript for a given run."""
    tr_root = Path(stimuli_root).parent / "transcripts"
    if film_id == "friends":
        p = tr_root / "friends" / "s6" / f"{run_id}.tsv"
    else:
        # movie10: transcript is prefixed with "movie10_"
        p = tr_root / "movie10" / film_id / f"movie10_{run_id}.tsv"
    return str(p) if p.exists() else None


def parse_transcript(tsv_path: str, T: int):
    """Parse transcript → (speech_coverage, word_rate) arrays at 1 Hz, length T."""
    all_onsets = []
    all_durs   = []

    with open(tsv_path) as f:
        next(f)                                 # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                onsets = ast.literal_eval(parts[-2])
                durs   = ast.literal_eval(parts[-1])
            except (ValueError, SyntaxError):
                continue
            all_onsets.extend(onsets)
            all_durs.extend(durs)

    speech_cov = np.zeros(T, dtype=np.float32)
    word_rate  = np.zeros(T, dtype=np.float32)

    for onset, dur in zip(all_onsets, all_durs):
        offset = onset + dur
        # word_rate: count onsets falling in each 1-s bin
        t0 = int(onset)
        if 0 <= t0 < T:
            word_rate[t0] += 1
        # speech_coverage: fractional overlap with each 1-s bin
        for t in range(max(0, int(onset)), min(T, int(offset) + 1)):
            ov = min(offset, t + 1.0) - max(onset, float(t))
            if ov > 0:
                speech_cov[t] += ov

    return np.clip(speech_cov, 0.0, 1.0), word_rate


# ------------------------------------------------------------------
#  Windowing
# ------------------------------------------------------------------

def windows_from_run(npz_path: Path, stimuli_root: str) -> list[dict]:
    d       = np.load(npz_path, allow_pickle=True)
    T       = int(len(d["time_s"]))
    run_id  = str(d["run_id"])
    film_id = str(d["film_id"])

    # A/V features from cache
    feats: dict[str, np.ndarray] = {f: d[f].astype(np.float64) for f in AV_FEATURES}

    # Transcript features
    tsv = find_transcript(run_id, film_id, stimuli_root)
    if tsv:
        sc, wr = parse_transcript(tsv, T)
        feats["speech_coverage"] = sc.astype(np.float64)
        feats["word_rate"]       = wr.astype(np.float64)
    else:
        feats["speech_coverage"] = np.full(T, np.nan)
        feats["word_rate"]       = np.full(T, np.nan)

    windows = []
    for start in range(0, T - WINDOW_S + 1, STRIDE_S):
        end = start + WINDOW_S
        w: dict = {
            "run_id": run_id, "film_id": film_id,
            "start_s": start, "end_s": end,
            "dist_to_start_s": start,
            "dist_to_end_s":   T - end,
            "has_transcript": tsv is not None,
        }

        for f in ALL_FEATURES:
            seg = feats[f][start:end]
            nan = bool(np.any(np.isnan(seg)))
            w[f"{f}_mean"] = None if nan else round(float(seg.mean()), 4)
            w[f"{f}_std"]  = None if nan else round(float(seg.std()),  4)

        # 4 blocks × 4 s
        blocks = []
        for b in range(N_BLOCKS):
            bs = start + b * BLOCK_S
            be = bs + BLOCK_S
            blk = {}
            for f in ALL_FEATURES:
                seg = feats[f][bs:be]
                blk[f] = None if bool(np.any(np.isnan(seg))) else round(float(seg.mean()), 4)
            blocks.append(blk)
        w["blocks"] = blocks

        windows.append(w)

    return windows


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main():
    cache   = Path(os.environ["CS_CACHE"]) / "cache_1hz"
    code    = Path(os.environ["CS_CODE"])
    stimuli = os.environ["CS_STIMULI"]

    npz_files = sorted(cache.glob("*.npz"))
    if not npz_files:
        sys.exit(f"No .npz in {cache} — run characterize_run.py first")

    all_wins: list[dict] = []
    n_no_tsv = 0
    for p in npz_files:
        ws = windows_from_run(p, stimuli)
        all_wins.extend(ws)
        has_t = ws[0]["has_transcript"] if ws else False
        if not has_t:
            n_no_tsv += 1
        print(f"  {p.stem:25s}  T={ws[-1]['end_s'] if ws else '?':>4}s  "
              f"→ {len(ws):4d} win  transcript={'✓' if has_t else '✗'}")

    out = code / "manifest" / "candidate_windows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n_windows": len(all_wins), "windows": all_wins}, f)

    films = {}
    for w in all_wins:
        films.setdefault(w["film_id"], 0)
        films[w["film_id"]] += 1

    print(f"\n{len(all_wins)} candidate windows → {out}")
    for film, n in sorted(films.items()):
        print(f"  {film:10s}: {n:5d}")
    if n_no_tsv:
        print(f"  ⚠ {n_no_tsv} runs missing transcript")
    print(f"  file size: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()