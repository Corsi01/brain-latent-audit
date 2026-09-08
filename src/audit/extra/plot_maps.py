#!/usr/bin/env python3
"""
plot_maps.py — rende le mappe a 364 valori sulla flat map 224x560.

Serve per l'ispezione anatomica: gli indici dei patch non dicono nulla finche'
non si vede dove cadono. frame_diff deve concentrarsi in occipitale, rms e
speech in temporale superiore.

Produce un PNG multi-pannello e stampa le coordinate di griglia dei patch di
picco.

Uso:
    python plot_maps.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH


def build_patch_map():
    from cortex_mae import transforms
    R = transforms._FLAT_RESAMPLER
    mask = np.asarray(R.mask_)
    m = mask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH * GW, -1)
    valid = m.any(axis=1)
    rank = np.cumsum(valid) - 1
    pop = np.full((H, W), -1, np.int32)
    for pi in np.flatnonzero(valid):
        gr, gc = divmod(pi, GW)
        pop[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH] = rank[pi]
    pop[~mask] = -1
    return pop, mask, valid


def render(v364, pop):
    img = np.full((H, W), np.nan, np.float32)
    ok = pop >= 0
    img[ok] = v364[pop[ok]]
    return img


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    sd = os.path.join(cache, "f0", "surface")
    ap.add_argument("--control", default=os.path.join(sd, "positive_control.npz"))
    ap.add_argument("--maps", default=os.path.join(
        cache, "f0", "maps", "participation_maps.npz"))
    ap.add_argument("--out", default=os.path.join(sd, "maps_flat.png"))
    args = ap.parse_args()

    pop, mask, valid = build_patch_map()
    valid_ids = np.flatnonzero(valid)

    ctrl = np.load(args.control)
    part = np.load(args.maps, allow_pickle=True)
    subs = [str(s) for s in part["subjects"]]
    part_group = np.mean([part[f"cos_{s}"] for s in subs], axis=0)

    panels = []
    for f in ("frame_diff", "luminance", "rms", "speech_coverage"):
        k = f"R_{f}"
        if k in ctrl:
            panels.append((f"encoding R: {f}", np.asarray(ctrl[k]), "RdBu_r", True))
    panels.append(("partecipazione (cos), media soggetti", part_group, "viridis", False))
    panels.append(("copertura corticale del patch", np.asarray(ctrl["cover"]),
                   "Greys", False))

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.6 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, v, cmap, sym) in zip(axes, panels):
        img = render(np.asarray(v, np.float64), pop)
        if sym:
            a = np.nanpercentile(np.abs(img), 99)
            im = ax.imshow(img, cmap=cmap, vmin=-a, vmax=a, interpolation="nearest")
        else:
            im = ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"figura salvata in {args.out}")

    # coordinate di griglia dei patch di picco
    print("\n=== PATCH DI PICCO: posizione sulla griglia 14 x 35 ===")
    print("   (riga, colonna) con riga 0 in alto, colonna 0 a sinistra")
    for f in ("frame_diff", "luminance", "rms", "speech_coverage"):
        k = f"R_{f}"
        if k not in ctrl:
            continue
        v = np.asarray(ctrl[k])
        top = np.argsort(-v)[:6]
        pos = [tuple(divmod(int(valid_ids[t]), GW)) for t in top]
        print(f"   {f:16s} {pos}")
    v = part_group
    top = np.argsort(-v)[:6]
    bot = np.argsort(v)[:6]
    print(f"   {'partecip. alta':16s} "
          f"{[tuple(divmod(int(valid_ids[t]), GW)) for t in top]}")
    print(f"   {'partecip. bassa':16s} "
          f"{[tuple(divmod(int(valid_ids[t]), GW)) for t in bot]}")

    np.save(os.path.join(os.path.dirname(args.out), "patch_of_pixel.npy"), pop)


if __name__ == "__main__":
    main()
