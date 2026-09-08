#!/usr/bin/env python3
"""
build_run_index.py — Discover all stimulus runs, write manifest/run_index.json.

Parses Friends S06 and Movie10 filenames from $CS_STIMULI.
Run once, then every other script reads the index.

Usage:
    source env.sh && python src/build_run_index.py
"""

import json
import os
import re
import struct
import sys
from pathlib import Path


def wav_duration(path: str) -> float | None:
    """Read WAV duration from header (no deps)."""
    try:
        with open(path, "rb") as f:
            riff, size, wave = struct.unpack("<4sI4s", f.read(12))
            if riff != b"RIFF" or wave != b"WAVE":
                return None
            while True:
                chunk_id, chunk_size = struct.unpack("<4sI", f.read(8))
                if chunk_id == b"fmt ":
                    fmt_data = f.read(chunk_size)
                    _, channels, sr, byterate, _, _ = struct.unpack_from("<HHIIHH", fmt_data)
                elif chunk_id == b"data":
                    return chunk_size / byterate
                else:
                    f.seek(chunk_size, 1)
    except Exception:
        return None


def discover_runs(stimuli_root: str) -> list[dict]:
    root = Path(stimuli_root)
    runs = []

    # --- Friends: friends/s6/friends_s06eNNP.{mkv,wav} ---
    friends_dir = root / "friends" / "s6"
    if friends_dir.exists():
        for mkv in sorted(friends_dir.glob("friends_s06e*.mkv")):
            m = re.match(r"friends_s06e(\d+)([a-z])", mkv.stem)
            if not m:
                continue
            ep, part = m.groups()
            wav = mkv.with_suffix(".wav")
            has_wav = wav.exists()
            runs.append({
                "run_id": mkv.stem,
                "film_id": "friends",
                "show": "friends",
                "episode": int(ep),
                "part": part,
                "video": str(mkv),
                "audio": str(wav) if has_wav else None,
                "audio_duration_s": wav_duration(str(wav)) if has_wav else None,
            })

    # --- Movie10: movie10/{film}/{film}NN.{mkv,wav} ---
    movie10_dir = root / "movie10"
    if movie10_dir.exists():
        for film_dir in sorted(movie10_dir.iterdir()):
            if not film_dir.is_dir():
                continue
            film = film_dir.name  # bourne, figures, life, wolf
            for mkv in sorted(film_dir.glob(f"{film}*.mkv")):
                m = re.match(rf"{re.escape(film)}(\d+)", mkv.stem)
                if not m:
                    continue
                wav = mkv.with_suffix(".wav")
                has_wav = wav.exists()
                runs.append({
                    "run_id": mkv.stem,
                    "film_id": film,
                    "show": "movie10",
                    "run_num": int(m.group(1)),
                    "video": str(mkv),
                    "audio": str(wav) if has_wav else None,
                    "audio_duration_s": wav_duration(str(wav)) if has_wav else None,
                })

    return runs


def main():
    stimuli = os.environ.get("CS_STIMULI")
    code = os.environ.get("CS_CODE")
    if not stimuli or not code:
        sys.exit("ERROR: source env.sh first (CS_STIMULI, CS_CODE must be set)")

    runs = discover_runs(stimuli)

    out_dir = Path(code) / "manifest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_index.json"
    with open(out_path, "w") as f:
        json.dump({"n_runs": len(runs), "runs": runs}, f, indent=2)

    # --- summary ---
    films: dict[str, list[str]] = {}
    missing_audio: list[str] = []
    total_dur = 0.0
    for r in runs:
        films.setdefault(r["film_id"], []).append(r["run_id"])
        if r["audio"] is None:
            missing_audio.append(r["run_id"])
        if r.get("audio_duration_s"):
            total_dur += r["audio_duration_s"]

    print(f"Found {len(runs)} runs ({total_dur/3600:.1f}h):")
    for film, rids in sorted(films.items()):
        print(f"  {film:10s}: {len(rids):3d} runs")
    if missing_audio:
        print(f"⚠ missing .wav: {missing_audio}")
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
