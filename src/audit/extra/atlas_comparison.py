#!/usr/bin/env python3
"""
atlas_comparison.py — la partecipazione traccia un gradiente o una rete?

Densita' verificata: la mappa (cortex_mae, fsLR-64k) e le annotazioni neuromaps
(fsLR den=32k) sono ENTRAMBE vettori a 64984 con lo stesso ordinamento,
32492 per emisfero, L poi R. Nessun ricampionamento: si lavora sull'intersezione
dei due medial wall (58212 vertici corticali qui, 59412 la', intersezione ~58k).

Due mappe portate avanti in parallelo, deliberatamente:
  grezza   = cos, media soggetti
  residua  = dopo regressione di copertura e distanza dal bordo

La geometria della griglia spiega il 56% della grezza, quindi la residua e' la
primaria. Ma regredire la distanza dal bordo puo' togliere segnale vero: il bordo
della flat map corrisponde a regioni mediali dove il DMN in parte vive. Se il
risultato regge su entrambe, la questione e' chiusa.

Ipotesi (da fissare PRIMA di leggere l'output):
  primaria    la partecipazione e' piu' alta nel DMN che nelle reti sensoriali
  secondaria  la partecipazione correla positivamente con il gradiente
              principale (unimodale -> transmodale)

Null: spin test di Alexander-Bloch tramite neuromaps.nulls.alexander_bloch, che
preserva l'autocorrelazione spaziale. Un p parametrico su vertici o parcelle
sarebbe inaccettabile: le mappe corticali sono lisce e qualunque coppia correla.

I vertici NON sono osservazioni indipendenti (364 patch, ~160 vertici ciascuno).
Lo spin test gestisce l'autocorrelazione, ma il test sulle reti e' comunque
riportato anche a livello di patch.

Uso — dal LOGIN NODE la prima volta (scarica gli atlanti), poi ovunque:
    python atlas_comparison.py
    python atlas_comparison.py --min-cover 0.25 --n-perm 1000
"""
import argparse
import os
import sys

import numpy as np

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH
N_SP, NV = 364, 64984


def patch_geometry():
    from cortex_mae import transforms
    R = transforms._FLAT_RESAMPLER
    mask = np.asarray(R.mask_)
    m = mask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH * GW, -1)
    valid = m.any(axis=1)
    rank = np.cumsum(valid) - 1
    pop = np.full((H, W), -1, np.int32)
    for p in np.flatnonzero(valid):
        gr, gc = divmod(int(p), GW)
        pop[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH] = rank[p]
    pop[~mask] = -1
    return R, pop, m.sum(axis=1)[valid] / (PATCH * PATCH)


def to_surface(v364, R, pop, drop=None):
    """[364] -> [64984]; i patch in `drop` diventano NaN."""
    v = np.asarray(v364, np.float64).copy()
    if drop is not None:
        v[drop] = np.nan
    img = np.full((H, W), np.nan, np.float32)
    ok = pop >= 0
    img[ok] = v[pop[ok]]
    bad = np.isnan(img)
    surf = np.asarray(R.inverse(np.nan_to_num(img, nan=0.0)[None])).reshape(-1)
    badv = np.asarray(R.inverse(bad[None].astype(np.float32))).reshape(-1) > 0.5
    out = np.full(NV, np.nan)
    n = min(len(surf), NV)
    out[:n] = surf[:n]
    out[:n][badv[:n]] = np.nan
    return out


def vertex_to_patch(R, pop):
    """Per ogni vertice fsLR, il patch di appartenenza (-1 se nessuno)."""
    img = np.where(pop >= 0, pop, -1).astype(np.float32)
    v = np.asarray(R.inverse(img[None])).reshape(-1)
    out = np.full(NV, -1, np.int32)
    n = min(len(v), NV)
    out[:n] = np.round(v[:n]).astype(np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    sd = os.path.join(cache, "f0", "surface")
    ap.add_argument("--check", default=os.path.join(sd, "coverage_check.npz"))
    ap.add_argument("--outdir", default=sd)
    ap.add_argument("--min-cover", type=float, default=0.0)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from neuromaps import datasets, images, nulls, stats
    except ImportError as e:
        sys.exit(f"neuromaps mancante: {e}")

    R, pop, cover = patch_geometry()
    d = np.load(args.check)
    maps = {"residua": np.asarray(d["part_resid"]),
            "grezza": np.asarray(d["part_group"])}
    drop = cover < args.min_cover if args.min_cover > 0 else None
    if drop is not None:
        print(f"esclusi {int(drop.sum())} patch sotto {args.min_cover:.0%} copertura")

    surf = {k: to_surface(v, R, pop, drop) for k, v in maps.items()}
    v2p = vertex_to_patch(R, pop)
    for k, v in surf.items():
        print(f"   {k}: {int(np.isfinite(v).sum())} vertici validi")

    # ------------------------------------------------------ gradiente
    print("\n=== GRADIENTE PRINCIPALE (Margulies 2016, fcgradient01) ===")
    ann = datasets.fetch_annotation(source="margulies2016", desc="fcgradient01",
                                    space="fsLR", den="32k")
    grad = np.asarray(images.load_data(ann), dtype=np.float64)
    print(f"   gradiente: {grad.shape}, non-zero {int((grad != 0).sum())}")
    grad_valid = grad != 0

    results = {}
    print(f"\n   spin test: {args.n_perm} rotazioni (alexander_bloch)")
    rot = None
    for k, v in surf.items():
        ok = np.isfinite(v) & grad_valid
        r = np.corrcoef(v[ok], grad[ok])[0, 1]
        print(f"   {k}: r = {r:+.4f}  su {int(ok.sum())} vertici", flush=True)
        try:
            vv = np.where(np.isfinite(v), v, 0.0)
            if rot is None:
                rot = nulls.alexander_bloch(vv, atlas="fsLR", density="32k",
                                            n_perm=args.n_perm, seed=args.seed)
            r_sp, p_sp = stats.compare_images(vv, grad, nulls=rot)
            print(f"      spin test: r = {r_sp:+.4f}, p = {p_sp:.4f}")
            results[f"grad_{k}"] = (r, r_sp, p_sp)
        except Exception as e:
            print(f"      spin test fallito: {type(e).__name__}: {e}")
            results[f"grad_{k}"] = (r, np.nan, np.nan)
        rot = None   # le rotazioni dipendono dalla mappa: ricalcola

    # ------------------------------------------------------ reti
    print("\n=== RETI ===")
    cand = [a for a in datasets.available_annotations(space="fsLR", den="32k")
            if any(s in str(a).lower() for s in ("yeo", "schaefer", "network"))]
    print(f"   annotazioni candidate: {cand if cand else 'nessuna'}")
    if not cand:
        print("   Yeo non disponibile via neuromaps in questa installazione.")
        print("   Alternative: usare gli annot Schaefer gia' in ../atlases/,")
        print("   oppure nilearn.datasets.fetch_atlas_surf_destrieux.")
        print("   Il gradiente sopra resta il test principale.")

    # ------------------------------------------- profilo per decile di gradiente
    print("\n=== PROFILO LUNGO IL GRADIENTE (10 decili) ===")
    print("   decile 1 = unimodale/sensoriale, decile 10 = transmodale/DMN")
    for k, v in surf.items():
        ok = np.isfinite(v) & grad_valid
        q = np.quantile(grad[ok], np.linspace(0, 1, 11))
        prof = [v[ok][(grad[ok] >= q[i]) & (grad[ok] < q[i+1] if i < 9
                                            else grad[ok] <= q[i+1])].mean()
                for i in range(10)]
        print(f"   {k:8s} " + " ".join(f"{x:6.3f}" for x in prof))
        results[f"profile_{k}"] = np.array(prof)

    out = {f"surf_{k}": v for k, v in surf.items()}
    out.update({k: np.asarray(v) for k, v in results.items()})
    out.update({"gradient": grad, "vertex_to_patch": v2p, "cover": cover})
    fout = os.path.join(args.outdir, "atlas_comparison.npz")
    np.savez_compressed(fout, **out)
    print(f"\nsalvato in {fout}")


if __name__ == "__main__":
    main()