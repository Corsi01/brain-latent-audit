#!/usr/bin/env python3
"""
characterize_pcs.py — che aspetto hanno PC3 e PC5, e a cosa somigliano?

Il contrasto Default vs sensorie dice che due componenti separano quelle reti,
non che aspetto abbiano ne' a quale organizzazione corticale corrispondano.
Questo script produce la caratterizzazione:

  1. mappe sulla flat map (PNG multi-pannello, tutte e 6 le componenti)
  2. profilo completo sulle 7 reti Yeo, non solo il contrasto DMN-sensorie
  3. correlazione con i primi gradienti di connettivita' (Margulies), con spin
     test: se una componente e' allineata al gradiente principale, e' la gamba
     neurobiologica
  4. correlazione con le mappe di encoding sensoriale gia' calcolate: una
     componente che tracciasse le aree evocate sarebbe sospetta
  5. geometria: le componenti sono contaminate da copertura e distanza dal bordo?

Nota di nomenclatura, importante per il paper: PC3 e PC5 non sono "la direzione
media". Sono componenti della VARIAZIONE FRA PATCH del vettore medio, dopo che
la media per patch e' stata rimossa. Oggetto diverso, nome diverso.

Uso:
    python characterize_pcs.py --n-perm 1000
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH
NV, N_SP, D, N_T = 64984, 364, 768, 4
N_PC = 6
NETS_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def patch_geometry():
    from cortex_mae import transforms
    R = transforms._FLAT_RESAMPLER
    mask = np.asarray(R.mask_)
    m = mask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH*GW, -1)
    valid = m.any(axis=1)
    rank = np.cumsum(valid) - 1
    pop = np.full((H, W), -1, np.int32)
    for p in np.flatnonzero(valid):
        gr, gc = divmod(int(p), GW)
        pop[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH] = rank[p]
    pop[~mask] = -1
    return R, pop


def render(v364, pop):
    img = np.full((H, W), np.nan, np.float32)
    ok = pop >= 0
    img[ok] = np.asarray(v364, np.float64)[pop[ok]]
    return img


def to_surface(v364, R, pop):
    s = np.asarray(R.inverse(
        np.nan_to_num(render(v364, pop), nan=0.0)[None])).reshape(-1)
    out = np.zeros(NV)
    n = min(len(s), NV)
    out[:n] = s[:n]
    return out


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    code = os.environ.get("CS_CODE", "..")
    sd = os.path.join(cache, "f0", "surface")
    ap.add_argument("--pcs", default=os.path.join(sd, "validate_pcs.npz"))
    ap.add_argument("--diag", default=os.path.join(sd, "diagnose_negative.npz"))
    ap.add_argument("--check", default=os.path.join(sd, "coverage_check.npz"))
    ap.add_argument("--outdir", default=sd)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-grad", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    R, pop = patch_geometry()
    pcs = np.load(args.pcs)
    scores = np.asarray(pcs["scores"])            # [364, 6]
    ev = np.asarray(pcs["ev"])
    diag = np.load(args.diag, allow_pickle=True)
    dom = np.asarray(diag["dom"]).astype(str)
    purity = np.asarray(diag["purity"])
    chk = np.load(args.check)
    cover, dist = np.asarray(chk["cover"]), np.asarray(chk["dist_edge"])

    # ------------------------------------------------------------ 1. figura
    ncol = 2
    fig, axes = plt.subplots(N_PC, 1, figsize=(13, 2.5*N_PC))
    for i in range(N_PC):
        img = render(scores[:, i], pop)
        a = np.nanpercentile(np.abs(img), 99)
        im = axes[i].imshow(img, cmap="RdBu_r", vmin=-a, vmax=a,
                            interpolation="nearest")
        axes[i].set_title(f"PC{i+1}  ({ev[i]:.1%} della varianza fra patch)",
                          fontsize=10, loc="left")
        axes[i].set_xticks([]); axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.02, pad=0.01)
    plt.tight_layout()
    fout = os.path.join(args.outdir, "pcs_flat.png")
    plt.savefig(fout, dpi=110, bbox_inches="tight")
    print(f"figura: {fout}")

    # ------------------------------------------- 2. profilo sulle 7 reti Yeo
    print("\n" + "="*72)
    print("2. PROFILO SULLE 7 RETI YEO  (z-score fra patch, per confrontabilita')")
    print("="*72)
    hdr = "   comp  " + " ".join(f"{n[:7]:>8s}" for n in NETS_ORDER)
    print(hdr)
    prof = {}
    for i in range(N_PC):
        v = scores[:, i]
        v = (v - v.mean()) / (v.std() + 1e-9)
        row = [v[dom == n].mean() if (dom == n).sum() >= 3 else np.nan
               for n in NETS_ORDER]
        prof[i] = np.array(row)
        star = "  *" if i in (2, 4) else ""
        print(f"   PC{i+1:<4d} " + " ".join(f"{x:+8.3f}" for x in row) + star)
    print("   Le reti sono in ordine canonico unimodale -> transmodale.")
    print("   Un profilo monotono indica un gradiente; uno a picco, una rete.")

    # ----------------------------------------------- 3. gradienti Margulies
    print("\n" + "="*72)
    print("3. CORRELAZIONE CON I GRADIENTI DI CONNETTIVITA'")
    print("="*72)
    from neuromaps import datasets, images, nulls, stats
    grads = {}
    for g in range(1, args.n_grad + 1):
        ann = datasets.fetch_annotation(source="margulies2016",
                                        desc=f"fcgradient{g:02d}",
                                        space="fsLR", den="32k")
        grads[g] = np.asarray(images.load_data(ann), dtype=np.float64)
    print(f"   {'comp':6s} " + " ".join(f"{'grad'+str(g):>16s}"
                                        for g in grads))
    corr = {}
    for i in range(N_PC):
        surf = to_surface(scores[:, i], R, pop)
        cells = []
        for g, gr in grads.items():
            ok = (gr != 0) & (surf != 0)
            r = float(np.corrcoef(surf[ok], gr[ok])[0, 1])
            p = np.nan
            if args.n_perm > 0 and i in (2, 4):
                try:
                    rot = nulls.alexander_bloch(surf, atlas="fsLR", density="32k",
                                                n_perm=args.n_perm, seed=args.seed)
                    _, p = stats.compare_images(surf, gr, nulls=rot)
                except Exception:
                    pass
            cells.append(f"{r:+.3f}" + (f" p={p:.3f}" if np.isfinite(p) else ""))
            corr[(i, g)] = (r, p)
        print(f"   PC{i+1:<4d} " + " ".join(f"{c:>16s}" for c in cells))
    print("   p calcolato (spin test) solo per PC3 e PC5, le componenti in esame.")

    # ------------------------------------- 4. contaminazione da encoding e geometria
    print("\n" + "="*72)
    print("4. CONTROLLI: le componenti tracciano le aree evocate o la geometria?")
    print("="*72)
    feats = [k[2:] for k in chk.files if k.startswith("R_")]
    print(f"   {'comp':6s} " + " ".join(f"{f[:9]:>10s}" for f in feats)
          + f" {'cover':>8s} {'dist':>8s}")
    for i in range(N_PC):
        v = scores[:, i]
        cells = [f"{np.corrcoef(v, np.asarray(chk['R_'+f]))[0,1]:+.3f}"
                 for f in feats]
        print(f"   PC{i+1:<4d} " + " ".join(f"{c:>10s}" for c in cells)
              + f" {np.corrcoef(v, cover)[0,1]:+8.3f}"
              f" {np.corrcoef(v, dist)[0,1]:+8.3f}")
    print("   |r| alto con le feature -> la componente traccia risposta evocata,")
    print("   non baseline. |r| alto con cover/dist -> contaminazione geometrica.")

    # ------------------------------------------------- 5. robustezza purezza
    print("\n" + "="*72)
    print("5. CONTRASTO Default - sensorie AI SOLI PATCH PURI")
    print("="*72)
    print(f"   {'soglia':8s} {'n':>5s} " + " ".join(f"{'PC'+str(i+1):>8s}"
                                                    for i in (2, 4)))
    for thr in (0.0, 0.6, 0.8, 0.9):
        sel = purity > thr
        cells = []
        for i in (2, 4):
            v = scores[sel, i]
            dd_ = dom[sel]
            a, b = v[dd_ == "Default"], v[np.isin(dd_, ["Vis", "SomMot"])]
            pooled = np.sqrt((a.var(ddof=1)*(len(a)-1) + b.var(ddof=1)*(len(b)-1))
                             / (len(a)+len(b)-2))
            cells.append(f"{(a.mean()-b.mean())/pooled:+.3f}")
        print(f"   >{thr:.1f}     {int(sel.sum()):5d} "
              + " ".join(f"{c:>8s}" for c in cells))

    np.savez_compressed(os.path.join(args.outdir, "characterize_pcs.npz"),
                        profiles=np.stack([prof[i] for i in range(N_PC)]),
                        nets=np.array(NETS_ORDER),
                        grad_r=np.array([[corr[(i, g)][0] for g in grads]
                                         for i in range(N_PC)]))
    print(f"\nsalvato in {args.outdir}/characterize_pcs.npz")


if __name__ == "__main__":
    main()
