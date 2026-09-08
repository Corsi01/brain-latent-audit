#!/usr/bin/env python3
"""
retrieval.py — The go/no-go test: is the CortexMAE latent stimulus-locked?

PRIMARY TEST — inter-subject same-clip retrieval.
  Query  = latent of clip c in subject A
  Match  = latent of clip c in subject B  (among all candidate clips in B)
  Logic  : intrinsic brain structure is idiosyncratic per subject. Two subjects
           align at the SAME timepoint only if the stimulus is driving them.
           Above-chance retrieval ⇒ the latent carries stimulus-locked signal.

SECONDARY — accuracy vs stimulus drive.
  If the latent is stimulus-locked (and not intrinsic), retrieval must SCALE
  with stimulus richness: high for SPEECH_DYNAMIC, at chance for BLACK.
  A flat curve would mean the retrieval is riding on intrinsic structure.

BASELINE — raw voxels (the flat map itself), same windows, same metric.
  Δ(latent − voxel) is the number that says whether the MAE isolates the
  evoked component or just passes it through.

Metrics: top-1, top-5, MRR, and PERCENTILE RANK (the headline — comparable
across conditions with different N; 0 = perfect, 0.5 = chance).

Usage:
    source env.sh && python src/retrieval.py --lag 4
    python src/retrieval.py --lag 4 --no-voxel          # skip voxel baseline
    python src/retrieval.py --lag 4 --exclude-same-run  # strict autocorrelation control
    python src/retrieval.py --lag 4 --block-level       # 4 s slot retrieval
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

N_BLOCKS   = 4
N_SPATIAL  = 364
EMB_DIM    = 768
NUM_FRAMES = 16
FLAT_DIM   = 77763


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------

def load_latents(lat_dir: Path, pooling: str = "spatial"):
    """
    Returns:
      X[subject][clip_id] = vector
      meta[clip_id] = dict(film_id, run_id, content_label, start_s)
      blocks[subject][clip_id] = (4, 768)   for block-level analysis
    """
    X      = defaultdict(dict)
    blocks = defaultdict(dict)
    meta   = {}

    files = sorted(lat_dir.glob("*.npz"))
    if not files:
        sys.exit(f"No latents in {lat_dir}")

    for f in files:
        d = np.load(f, allow_pickle=True)
        sub     = str(d["subject"])
        clip_id = str(d["clip_id"])
        p = d["patch_embeds"]                       # (1456, 768) = 4 × 364

        if p.shape[0] != N_BLOCKS * N_SPATIAL:
            # some clips may have a different valid-patch count; reshape by inference
            n_sp = p.shape[0] // N_BLOCKS
            r = p.reshape(N_BLOCKS, n_sp, EMB_DIM)
        else:
            r = p.reshape(N_BLOCKS, N_SPATIAL, EMB_DIM)

        # ---- spatial pooling: mean over cortical patches → (4, 768) ----
        if pooling == "spatial":
            v = r.mean(axis=1)                      # (4, 768)
        elif pooling == "full":
            v = r.mean(axis=(0, 1))[None, :]        # (1, 768) — no temporal structure
        elif pooling == "flat":
            v = r.reshape(N_BLOCKS, -1)             # (4, n_sp*768) — no pooling
        else:
            raise ValueError(pooling)

        blocks[sub][clip_id] = v.astype(np.float32)
        X[sub][clip_id]      = v.ravel().astype(np.float32)

        if clip_id not in meta:
            meta[clip_id] = {
                "film_id":       str(d["film_id"]),
                "run_id":        str(d["run_id"]),
                "content_label": str(d["content_label"]),
                "start_s":       int(d["start_s"]),
            }

    return dict(X), dict(blocks), meta


def load_voxels(code: Path, cache: Path, lag: int, clip_ids: set):
    """
    Raw flat-map baseline: same windows, same lag, temporally binned to 4 blocks.

    Iterates RUN BY RUN so only one run (~218 MB) is resident at a time.
    Holding every run of a subject at once would peak at >10 GB and get OOM-killed.
    """
    with open(code / "manifest" / "clip_manifest.json") as f:
        clips = {c["clip_id"]: c for c in json.load(f)["clips"]}

    fdir = cache / "fmri_flat_1hz"
    V = defaultdict(dict)

    # group the requested clips by run once
    by_run = defaultdict(list)
    for clip_id in clip_ids:
        c = clips.get(clip_id)
        if c is not None:
            by_run[c["run_id"]].append(c)

    subs = sorted({p.stem.split("_", 1)[0] for p in fdir.glob("*.npy")})
    for sub in subs:
        for run_id, run_clips in by_run.items():
            npy = fdir / f"{sub}_{run_id}.npy"
            if not npy.exists():
                continue

            fmri = np.load(npy).astype(np.float32)      # (T, 77763) — one run only

            for c in run_clips:
                # same lag convention as extract_latents: fMRI window = stim window + LAG
                t0 = c["start_s"] + lag
                t1 = t0 + NUM_FRAMES
                if t1 > fmri.shape[0]:
                    continue
                w = fmri[t0:t1]                                    # (16, 77763)
                w = w.reshape(N_BLOCKS, 4, FLAT_DIM).mean(axis=1)  # (4, 77763)
                V[sub][c["clip_id"]] = w.ravel().astype(np.float32)

            del fmri                                     # free before the next run

    return dict(V)


# ---------------------------------------------------------------------------
#  Core retrieval
# ---------------------------------------------------------------------------

def cosine_ranks(Q: np.ndarray, C: np.ndarray, true_idx: np.ndarray,
                 forbid: np.ndarray | None = None):
    """
    Q (nq, d) queries, C (nc, d) candidates, true_idx (nq,) correct candidate.
    forbid (nq, nc) bool: candidates to exclude (e.g. same-run, sibling slots).
    Returns ranks (0 = best) and the number of valid candidates per query.
    """
    Qn = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8, None)
    Cn = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-8, None)
    S = Qn @ Cn.T                                   # (nq, nc)

    if forbid is not None:
        S = np.where(forbid, -np.inf, S)

    true_s = S[np.arange(len(Q)), true_idx]
    valid  = np.isfinite(S)
    ranks  = ((S > true_s[:, None]) & valid).sum(axis=1)
    n_cand = valid.sum(axis=1)
    return ranks, n_cand


def evaluate(X: dict, meta: dict, exclude_same_run: bool = False):
    """Inter-subject same-clip retrieval over all ordered subject pairs."""
    subs = sorted(X.keys())
    all_ranks, all_ncand, all_clips = [], [], []

    for A in subs:
        for B in subs:
            if A == B:
                continue
            shared = sorted(set(X[A]) & set(X[B]))
            if len(shared) < 10:
                continue

            Q = np.stack([X[A][c] for c in shared])
            C = np.stack([X[B][c] for c in shared])
            true_idx = np.arange(len(shared))

            forbid = None
            if exclude_same_run:
                runs = np.array([meta[c]["run_id"] for c in shared])
                # forbid candidates from the same run, EXCEPT the true match
                forbid = runs[:, None] == runs[None, :]
                forbid[true_idx, true_idx] = False

            ranks, ncand = cosine_ranks(Q, C, true_idx, forbid)
            all_ranks.append(ranks)
            all_ncand.append(ncand)
            all_clips.append(np.array(shared))

    if not all_ranks:
        sys.exit("No shared clips across subjects")

    return (np.concatenate(all_ranks),
            np.concatenate(all_ncand),
            np.concatenate(all_clips))


def summarize(ranks, ncand, label=""):
    pct = ranks / np.clip(ncand - 1, 1, None)       # 0 = perfect, 0.5 = chance
    return {
        "label":   label,
        "n":       len(ranks),
        "top1":    float((ranks == 0).mean()),
        "top5":    float((ranks < 5).mean()),
        "mrr":     float((1.0 / (ranks + 1)).mean()),
        "pct":     float(pct.mean()),
        "chance1": float(np.mean(1.0 / np.clip(ncand, 1, None))),
    }


def print_row(s):
    star = ""
    if s["pct"] < 0.45:
        star = " ***" if s["pct"] < 0.35 else " *"
    print(f"  {s['label']:20s} n={s['n']:5d}  "
          f"top1={s['top1']:.3f} (chance {s['chance1']:.3f})  "
          f"top5={s['top5']:.3f}  MRR={s['mrr']:.3f}  "
          f"pct-rank={s['pct']:.3f}{star}")


# ---------------------------------------------------------------------------
#  Block-level (4 s slot) retrieval
# ---------------------------------------------------------------------------

def evaluate_blocks(blocks: dict, meta: dict):
    """
    Retrieval on individual 4 s slots. Sibling slots from the SAME 16 s clip
    share encoder context, so they are excluded from the candidate pool —
    otherwise the top match is trivially the neighbouring slot and the
    accuracy is context leakage, not stimulus locking.
    """
    subs = sorted(blocks.keys())
    all_ranks, all_ncand = [], []

    for A in subs:
        for B in subs:
            if A == B:
                continue
            shared = sorted(set(blocks[A]) & set(blocks[B]))
            if len(shared) < 10:
                continue

            keys, Qs, Cs = [], [], []
            for c in shared:
                for b in range(N_BLOCKS):
                    keys.append((c, b))
                    Qs.append(blocks[A][c][b])
                    Cs.append(blocks[B][c][b])

            Q = np.stack(Qs); C = np.stack(Cs)
            true_idx = np.arange(len(keys))

            clip_of = np.array([k[0] for k in keys])
            forbid = clip_of[:, None] == clip_of[None, :]   # siblings, same clip
            forbid[true_idx, true_idx] = False              # keep the true match

            ranks, ncand = cosine_ranks(Q, C, true_idx, forbid)
            all_ranks.append(ranks)
            all_ncand.append(ncand)

    return np.concatenate(all_ranks), np.concatenate(all_ncand)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, required=True)
    p.add_argument("--pooling", default="spatial", choices=["spatial", "full", "flat"])
    p.add_argument("--no-voxel", action="store_true", help="skip raw-voxel baseline")
    p.add_argument("--exclude-same-run", action="store_true")
    p.add_argument("--block-level", action="store_true")
    p.add_argument("--n-perm", type=int, default=200)
    args = p.parse_args()

    code  = Path(os.environ["CS_CODE"])
    cache = Path(os.environ["CS_CACHE"])
    lat_dir = cache / f"latents_lag{args.lag}"

    print(f"lag={args.lag}  pooling={args.pooling}  "
          f"exclude_same_run={args.exclude_same_run}\n")

    X, blocks, meta = load_latents(lat_dir, pooling=args.pooling)
    subs = sorted(X.keys())
    n_clips = len(meta)
    print(f"{len(subs)} subjects × {n_clips} clips  →  "
          f"vector dim = {len(next(iter(X[subs[0]].values())))}\n")

    # ================= PRIMARY: inter-subject same-clip =================
    print("=" * 78)
    print("PRIMARY — inter-subject same-clip retrieval  (CortexMAE latent)")
    print("=" * 78)

    ranks, ncand, clips = evaluate(X, meta, args.exclude_same_run)
    overall = summarize(ranks, ncand, "ALL CLIPS")
    print_row(overall)

    # permutation null on the percentile rank
    pct = ranks / np.clip(ncand - 1, 1, None)
    rng = np.random.default_rng(0)
    null = [rng.permutation(pct).mean() for _ in range(args.n_perm)]  # mean is invariant
    # a real null: random ranks uniform over candidates
    null_pct = [rng.integers(0, ncand).mean() / np.clip(ncand - 1, 1, None).mean()
                for _ in range(args.n_perm)]
    print(f"\n  empirical chance pct-rank ≈ {np.mean(null_pct):.3f} "
          f"± {np.std(null_pct):.3f}   (theoretical 0.500)")
    z = (0.5 - overall["pct"]) / max(np.std(null_pct), 1e-6)
    print(f"  latent is {z:.1f} SD better than chance")

    # ================= BASELINE: raw voxels =================
    vox_overall = None
    if not args.no_voxel:
        print("\n" + "=" * 78)
        print("BASELINE — same retrieval on RAW VOXELS (flat map)")
        print("=" * 78)
        V = load_voxels(code, cache, args.lag, set(meta.keys()))
        vranks, vncand, _ = evaluate(V, meta, args.exclude_same_run)
        vox_overall = summarize(vranks, vncand, "ALL CLIPS (voxel)")
        print_row(vox_overall)

        d = vox_overall["pct"] - overall["pct"]     # positive = latent better
        print(f"\n  Δ pct-rank (voxel − latent) = {d:+.3f}   "
              f"{'→ latent BETTER' if d > 0 else '→ voxel better'}")

    # ================= DRIVE CURVE =================
    print("\n" + "=" * 78)
    print("DRIVE CURVE — retrieval by content category")
    print("  expected if stimulus-locked: BLACK/LOW_DRIVE ≈ chance, rising with drive")
    print("=" * 78)

    order = ["BLACK", "LOW_DRIVE", "AUDIO_NOSPEECH", "VISUAL_NOSPEECH",
             "SPEECH_STATIC", "SPEECH_DYNAMIC"]
    by_cat = defaultdict(lambda: ([], []))
    for r, nc, c in zip(ranks, ncand, clips):
        lab = meta[c]["content_label"]
        by_cat[lab][0].append(r)
        by_cat[lab][1].append(nc)

    for cat in order:
        if cat not in by_cat:
            continue
        r, nc = by_cat[cat]
        print_row(summarize(np.array(r), np.array(nc), cat))

    # ================= FILM CONTROL =================
    print("\n" + "=" * 78)
    print("FILM CONTROL — does the signal survive within a single film?")
    print("=" * 78)
    films = defaultdict(lambda: ([], []))
    for r, nc, c in zip(ranks, ncand, clips):
        f = meta[c]["film_id"]
        films[f][0].append(r)
        films[f][1].append(nc)
    for f in sorted(films):
        r, nc = films[f]
        print_row(summarize(np.array(r), np.array(nc), f))

    # ================= BLOCK LEVEL =================
    if args.block_level:
        print("\n" + "=" * 78)
        print("BLOCK LEVEL — 4 s slots  (sibling slots excluded from candidates)")
        print("=" * 78)
        br, bn = evaluate_blocks(blocks, meta)
        print_row(summarize(br, bn, "4s SLOTS"))

    # ================= VERDICT =================
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    lat_ok = overall["pct"] < 0.45
    print(f"  above-chance same-clip retrieval : {'YES' if lat_ok else 'NO'} "
          f"(pct-rank {overall['pct']:.3f} vs 0.500)")
    if "BLACK" in by_cat and "SPEECH_DYNAMIC" in by_cat:
        b = summarize(np.array(by_cat["BLACK"][0]), np.array(by_cat["BLACK"][1]))["pct"]
        s = summarize(np.array(by_cat["SPEECH_DYNAMIC"][0]),
                      np.array(by_cat["SPEECH_DYNAMIC"][1]))["pct"]
        print(f"  drive gradient (BLACK − SPEECH_DYNAMIC) : {b - s:+.3f} "
              f"{'✓ correct direction' if b > s else '✗ FLAT/INVERTED — suspicious'}")
    if vox_overall:
        print(f"  latent vs raw voxels : "
              f"{vox_overall['pct'] - overall['pct']:+.3f}")
    print()
    print("  Necessary condition for using frozen CortexMAE as an encoding target:")
    print("  above-chance retrieval AND a drive gradient. A latent at chance, or flat")
    print("  across drive, means the target carries no isolable evoked component.")


if __name__ == "__main__":
    main()
