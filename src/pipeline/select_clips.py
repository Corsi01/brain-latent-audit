#!/usr/bin/env python3
"""
select_clips.py — Classify + select clips using transcript-based speech
                   detection and video/audio features.

Categories are a 2-axis factorial (speech × motion) + controls:
  BLACK            near-zero luminance (control, retrieval ≈ chance)
  LOW_DRIVE        no speech, low motion, quiet audio (near-rest)
  AUDIO_NOSPEECH   no speech but audible content (music/SFX)
  VISUAL_NOSPEECH  no speech, high visual motion (action w/o dialogue)
  SPEECH_STATIC    dense speech, low motion (talking heads)
  SPEECH_DYNAMIC   dense speech, high motion (dialogue + action)

These map to a drive gradient for the retrieval-vs-drive curve:
  BLACK ≈ LOW_DRIVE < AUDIO/VISUAL_NOSPEECH < SPEECH_STATIC < SPEECH_DYNAMIC

Reads:  $CS_CODE/manifest/candidate_windows.json
Writes: $CS_CODE/manifest/clip_manifest.json

Usage:
    source env.sh && python src/select_clips.py
    python src/select_clips.py --target 40 --sep 90
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

# =====================================================================
#  CONFIG — adjust and re-run (< 1 s)
# =====================================================================
CONFIG = {
    # Constraints
    "edge_margin_s":        10,
    "min_separation_s":     60,
    "target_per_category":  30,
    "min_films_per_cat":     2,

    # BLACK
    "black_lum_max":        15.0,
    "black_lum_std_max":     5.0,

    # Speech axis (from transcript ground truth)
    "speech_high":           0.30,   # >30% coverage → "has speech"
    "speech_low":            0.10,   # <10% coverage → "no speech"

    # Motion axis (percentile-based, computed from data)
    "low_motion_pct":        25,     # frame_diff below P25
    "high_motion_pct":       50,     # frame_diff above P50

    # Audio level (for LOW_DRIVE vs AUDIO_NOSPEECH)
    "quiet_rms_pct":         25,     # rms below P25 → quiet
}

CATEGORY_ORDER = [
    "BLACK",
    "LOW_DRIVE",
    "AUDIO_NOSPEECH",
    "VISUAL_NOSPEECH",
    "SPEECH_STATIC",
    "SPEECH_DYNAMIC",
]


# =====================================================================
#  Stats
# =====================================================================

def print_stats(windows):
    import numpy as np

    feats = [f"{f}_mean" for f in (
        "luminance", "frame_diff", "rms",
        "spectral_flatness", "speech_coverage", "word_rate")]

    print("=" * 76)
    print(f"{'feature':>28s}   {'P5':>8s} {'P25':>8s} {'P50':>8s} {'P75':>8s} {'P95':>8s}")
    print("-" * 76)
    for f in feats:
        vals = [w[f] for w in windows if w.get(f) is not None]
        if not vals:
            print(f"{f:>28s}   {'(no data)':>8s}")
            continue
        a = np.array(vals)
        ps = np.percentile(a, [5, 25, 50, 75, 95])
        print(f"{f:>28s}   {ps[0]:8.4f} {ps[1]:8.4f} {ps[2]:8.4f} {ps[3]:8.4f} {ps[4]:8.4f}")
    print("=" * 76)


# =====================================================================
#  Classification
# =====================================================================

def classify_windows(windows, cfg):
    import numpy as np

    # Data-driven percentile thresholds
    fd_vals  = [w["frame_diff_mean"] for w in windows if w["frame_diff_mean"] is not None]
    rms_vals = [w["rms_mean"]        for w in windows if w["rms_mean"]        is not None]

    fd_arr  = np.array(fd_vals)  if fd_vals  else np.array([0.0])
    rms_arr = np.array(rms_vals) if rms_vals else np.array([0.0])

    fd_lo  = float(np.percentile(fd_arr,  cfg["low_motion_pct"]))
    fd_hi  = float(np.percentile(fd_arr,  cfg["high_motion_pct"]))
    rms_lo = float(np.percentile(rms_arr, cfg["quiet_rms_pct"]))

    print(f"\nDerived thresholds:  frame_diff P{cfg['low_motion_pct']}={fd_lo:.2f}  "
          f"P{cfg['high_motion_pct']}={fd_hi:.2f}  |  rms P{cfg['quiet_rms_pct']}={rms_lo:.4f}")

    for w in windows:
        lum   = w.get("luminance_mean")
        lum_s = w.get("luminance_std")
        fd    = w.get("frame_diff_mean")
        rms   = w.get("rms_mean")
        sp    = w.get("speech_coverage_mean")

        # --- defaults ---
        w["_label"]    = "OTHER"
        w["_strength"] = 0.0

        if lum is None:
            w["_label"] = "SKIP"
            continue

        # --- BLACK ---
        if lum < cfg["black_lum_max"] and (lum_s or 0) < cfg["black_lum_std_max"]:
            w["_label"]    = "BLACK"
            w["_strength"] = cfg["black_lum_max"] - lum
            continue

        # need transcript for speech categories
        if sp is None:
            # fallback: classify on motion only
            if fd is not None and fd > fd_hi:
                w["_label"]    = "VISUAL_NOSPEECH"
                w["_strength"] = fd
            continue

        no_speech  = sp < cfg["speech_low"]
        has_speech = sp > cfg["speech_high"]

        if no_speech:
            if fd is not None and fd < fd_lo and rms is not None and rms < rms_lo:
                w["_label"]    = "LOW_DRIVE"
                w["_strength"] = 1.0 / (fd + rms + 1e-6)    # quieter+stiller = stronger
            elif rms is not None and rms >= rms_lo and (fd is None or fd < fd_hi):
                w["_label"]    = "AUDIO_NOSPEECH"
                w["_strength"] = rms                          # louder = stronger example
            elif fd is not None and fd >= fd_hi:
                w["_label"]    = "VISUAL_NOSPEECH"
                w["_strength"] = fd
            # else OTHER (gap zone)

        elif has_speech:
            if fd is not None and fd < fd_lo:
                w["_label"]    = "SPEECH_STATIC"
                w["_strength"] = sp                           # more speech = stronger
            elif fd is not None and fd >= fd_hi:
                w["_label"]    = "SPEECH_DYNAMIC"
                w["_strength"] = sp + fd / 100.0              # both axes
            # else OTHER (mid-motion gap)

        # else: 0.10 ≤ speech ≤ 0.30 → ambiguous → OTHER

    return windows


def print_classification(windows):
    cats = defaultdict(list)
    for w in windows:
        cats[w["_label"]].append(w)

    print(f"\n{'category':>20s}  {'count':>6s}  {'films'}")
    print("-" * 70)
    for cat in sorted(cats.keys()):
        ws = cats[cat]
        films = sorted(set(w["film_id"] for w in ws))
        parts = [f"{f}({sum(1 for w in ws if w['film_id']==f)})" for f in films]
        print(f"{cat:>20s}  {len(ws):6d}  {', '.join(parts)}")
    print()


# =====================================================================
#  Selection (round-robin, global separation)
# =====================================================================

def select_clips(windows, cfg):
    target  = cfg["target_per_category"]
    min_sep = cfg["min_separation_s"]
    edge    = cfg["edge_margin_s"]

    pool = [w for w in windows
            if w["_label"] in CATEGORY_ORDER
            and w.get("dist_to_start_s", 999) >= edge]

    by_cat = {}
    for cat in CATEGORY_ORDER:
        by_cat[cat] = sorted(
            [w for w in pool if w["_label"] == cat],
            key=lambda w: w.get("_strength", 0), reverse=True)

    idx      = {c: 0 for c in CATEGORY_ORDER}
    selected = {c: [] for c in CATEGORY_ORDER}
    blocked  = {}          # run_id → [start_s, ...]
    active   = set(CATEGORY_ORDER)

    while active:
        progress = False
        for cat in CATEGORY_ORDER:
            if cat not in active:
                continue
            if len(selected[cat]) >= target:
                active.discard(cat)
                continue
            cands = by_cat[cat]
            i = idx[cat]
            while i < len(cands):
                w = cands[i]; i += 1
                run, st = w["run_id"], w["start_s"]
                if run in blocked and any(abs(st - t) < min_sep for t in blocked[run]):
                    continue
                
                film = w["film_id"]
                film_count = sum(1 for c in selected[cat] if c["film_id"] == film)
                if film_count >= max(target // 2, 1):
                    continue
                
                selected[cat].append(w)
                blocked.setdefault(run, []).append(st)
                idx[cat] = i
                progress = True
                break
            else:
                idx[cat] = i
                active.discard(cat)
        if not progress:
            break

    return selected


def print_selection(selected, cfg):
    total = 0
    print(f"\n{'category':>20s}  {'sel':>5s}  {'films'}")
    print("-" * 70)
    for cat in CATEGORY_ORDER:
        clips = selected[cat]
        films = sorted(set(c["film_id"] for c in clips))
        n = len(clips); total += n
        flag = " ⚠ <2 films" if len(films) < cfg["min_films_per_cat"] and n > 0 else ""
        parts = [f"{f}({sum(1 for c in clips if c['film_id']==f)})" for f in films]
        print(f"{cat:>20s}  {n:5d}  {', '.join(parts)}{flag}")
    print(f"{'TOTAL':>20s}  {total:5d}")


# =====================================================================
#  Manifest
# =====================================================================

def build_manifest(selected):
    clips = []
    for cat, ws in selected.items():
        for w in ws:
            clips.append({
                "clip_id":        f"{w['run_id']}_t{w['start_s']:04d}",
                "run_id":         w["run_id"],
                "film_id":        w["film_id"],
                "start_s":        w["start_s"],
                "end_s":          w["end_s"],
                "content_label":  cat,
                "drive_vector": {
                    "luminance":        w["luminance_mean"],
                    "frame_diff":       w["frame_diff_mean"],
                    "rms":              w["rms_mean"],
                    "speech_coverage":  w["speech_coverage_mean"],
                    "word_rate":        w["word_rate_mean"],
                    "spectral_flatness": w["spectral_flatness_mean"],
                },
                "drive_std": {
                    "luminance":        w["luminance_std"],
                    "frame_diff":       w["frame_diff_std"],
                    "rms":              w["rms_std"],
                    "speech_coverage":  w["speech_coverage_std"],
                },
                "blocks":           w["blocks"],
                "dist_to_start_s":  w.get("dist_to_start_s", 0),
            })

    summary = {cat: len(ws) for cat, ws in selected.items() if ws}
    return {"n_clips": len(clips), "categories": summary, "clips": clips}


# =====================================================================
#  Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, help="clips per category")
    p.add_argument("--sep",    type=int, help="min separation (s)")
    p.add_argument("--edge",   type=int, help="edge margin (s)")
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.target: cfg["target_per_category"] = args.target
    if args.sep:    cfg["min_separation_s"]    = args.sep
    if args.edge:   cfg["edge_margin_s"]       = args.edge

    code = Path(os.environ["CS_CODE"])
    inp  = code / "manifest" / "candidate_windows.json"
    if not inp.exists():
        sys.exit(f"Not found: {inp}\nRun build_windows.py first.")

    with open(inp) as f:
        data = json.load(f)
    windows = data["windows"]
    print(f"Loaded {len(windows)} candidate windows\n")

    print_stats(windows)
    classify_windows(windows, cfg)
    print_classification(windows)

    selected = select_clips(windows, cfg)
    print_selection(selected, cfg)

    manifest = build_manifest(selected)
    out = code / "manifest" / "clip_manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n→ {out}  ({manifest['n_clips']} clips)")


if __name__ == "__main__":
    main()