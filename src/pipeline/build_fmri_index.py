#!/usr/bin/env python3
"""
build_fmri_index.py — Map (subject, run_id) → fMRI .nii.gz path.

Writes: $CS_CODE/manifest/fmri_index.json

Only indexes runs that actually appear in clip_manifest.json.
Movie10 repeats (run-1 / run-2): keeps run-1 by default.

Usage:
    source env.sh && python src/build_fmri_index.py
"""

import json, os, re, sys
from pathlib import Path
from collections import defaultdict


def task_from_run_id(run_id: str) -> str:
    """manifest run_id → BIDS task label."""
    if run_id.startswith("friends_"):
        return run_id.split("friends_")[1]     # friends_s06e01a → s06e01a
    return run_id                              # bourne01 → bourne01


def main():
    code  = Path(os.environ["CS_CODE"])
    fmri  = Path(os.environ["CS_FMRI"])        # .../download

    with open(code / "manifest" / "clip_manifest.json") as f:
        clips = json.load(f)["clips"]

    needed = sorted({c["run_id"] for c in clips})
    task2run = {task_from_run_id(r): r for r in needed}
    print(f"{len(needed)} runs referenced by the manifest\n")

    entries = []
    found = defaultdict(set)

    for dataset in ("friends.fmriprep", "movie10.fmriprep"):
        root = fmri / dataset
        if not root.exists():
            print(f"⚠ missing {root}")
            continue
        for nii in root.rglob("*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"):
            name = nii.name
            m_sub  = re.search(r"(sub-\d+)", name)
            m_task = re.search(r"task-([A-Za-z0-9]+)", name)
            if not (m_sub and m_task):
                continue
            task = m_task.group(1)
            if task not in task2run:
                continue                      # run not selected
            m_rep = re.search(r"_run-(\d+)_", name)
            rep = int(m_rep.group(1)) if m_rep else 1
            if rep != 1:
                continue                      # movie10 repeat → keep run-1 only

            sub = m_sub.group(1)
            run_id = task2run[task]
            entries.append({
                "subject": sub,
                "run_id":  run_id,
                "task":    task,
                "path":    str(nii),
            })
            found[run_id].add(sub)

    entries.sort(key=lambda e: (e["subject"], e["run_id"]))
    out = code / "manifest" / "fmri_index.json"
    with open(out, "w") as f:
        json.dump({"n_entries": len(entries), "entries": entries}, f, indent=2)

    subs = sorted({e["subject"] for e in entries})
    print(f"subjects: {subs}")
    print(f"{len(entries)} (subject, run) pairs → {out}\n")

    # coverage: which runs are missing which subjects?
    incomplete = {r: sorted(set(subs) - found[r]) for r in needed if set(subs) - found[r]}
    if incomplete:
        print(f"⚠ {len(incomplete)}/{len(needed)} runs incomplete across subjects:")
        for r, missing in list(incomplete.items())[:10]:
            print(f"    {r:20s} missing {missing}")
        print("  → clips in these runs will have fewer than 4 subjects for retrieval")
    else:
        print("✓ all selected runs present for all subjects")


if __name__ == "__main__":
    main()
