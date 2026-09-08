#!/usr/bin/env python3
"""
patch_to_surface.py — porta la mappa a 364 valori sui vertici fsLR-64k.

Catena, tutta gia' presente in cortex_mae.nisc.FlatResampler:

    patch (griglia 14 x 35, 16x16 pixel)
      -> pixel validi nella maschera 224 x 560   (77763)
      -> vertici fsLR-64k                        (58212 corticali su 64984)

Il resampler espone `inverse()`, cioe' flat -> superficie, che e' esattamente la
direzione che serve. Non si ricostruisce nulla: si usa il loro codice.

L'unica incognita e' quali 364 dei 490 patch siano validi e in che ordine. Lo
script la risolve empiricamente cercando la soglia di copertura che restituisce
esattamente 364 patch, e verifica il risultato con un round-trip.

Uso:
    python patch_to_surface.py --maps $CS_CACHE/f0/maps/participation_maps.npz
"""
import argparse
import os

import numpy as np

H, W = 224, 560
PATCH = 16
GH, GW = H // PATCH, W // PATCH          # 14 x 35 = 490
N_EXPECTED = 364
FSLR_V = 64984


def patchify_mask(mask):
    """[224, 560] bool -> [490, 256] bool, ordine row-major sulla griglia."""
    m = mask.reshape(GH, PATCH, GW, PATCH)
    m = m.transpose(0, 2, 1, 3).reshape(GH * GW, PATCH * PATCH)
    return m


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    ap.add_argument("--maps", default=os.path.join(
        cache, "f0", "maps", "participation_maps.npz"))
    ap.add_argument("--outdir", default=os.path.join(cache, "f0", "surface"))
    ap.add_argument("--kind", default="cos",
                    help="quale mappa proiettare: cos | proj | norm | proj_raw")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from cortex_mae import transforms
    R = transforms._FLAT_RESAMPLER
    mask = np.asarray(R.mask_)
    point_mask = np.asarray(R.point_mask_)
    print(f"griglia {mask.shape}  pixel validi {mask.sum()}")
    print(f"vertici fsLR {point_mask.shape[0]}  corticali {point_mask.sum()}")

    # ------------------------------------------------------- 1. selezione patch
    print("\n=== 1. QUALI 364 PATCH SU 490 ===")
    pm = patchify_mask(mask)
    cover = pm.sum(axis=1) / (PATCH * PATCH)
    print(f"   patch con almeno un pixel valido: {(cover > 0).sum()}")
    for thr in (0.0, 0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        n = int((cover > thr).sum())
        flag = "  <== 364" if n == N_EXPECTED else ""
        print(f"   copertura > {thr:.3f}: {n:3d} patch{flag}")

    cands = [t for t in np.unique(np.round(cover, 6)) if (cover > t).sum() == N_EXPECTED]
    if not cands:
        raise SystemExit(
            "Nessuna soglia di copertura da esattamente 364 patch.\n"
            "La selezione non e' per soglia: cerca in models_mae.py come vengono\n"
            "scelti i patch validi (probabilmente una patch_mask esplicita).")
    thr = float(cands[0])
    valid = cover > thr
    print(f"\n   soglia adottata: copertura > {thr:.4f}  ->  {valid.sum()} patch")
    print(f"   ordine assunto: row-major sulla griglia {GH} x {GW},")
    print(f"   restringendo ai patch validi. VERIFICARE contro models_mae.py.")

    # ------------------------------------------------- 2. mappa patch -> pixel
    patch_of_pixel = np.full((H, W), -1, dtype=np.int32)
    rank = np.cumsum(valid) - 1
    for p in range(GH * GW):
        if not valid[p]:
            continue
        gr, gc = divmod(p, GW)
        patch_of_pixel[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH] = rank[p]
    patch_of_pixel[~mask] = -1
    covered = (patch_of_pixel >= 0).sum()
    print(f"\n=== 2. PIXEL COPERTI DAI PATCH VALIDI ===")
    print(f"   {covered} / {mask.sum()}  ({100*covered/mask.sum():.1f}%)")
    print(f"   i pixel scoperti stanno nei {int((~valid).sum())} patch a bassa copertura")

    # -------------------------------------------- 3. proiezione sulla superficie
    print(f"\n=== 3. PROIEZIONE SUI VERTICI fsLR ===")
    data = np.load(args.maps, allow_pickle=True)
    subs = [str(s) for s in data["subjects"]]
    print(f"   soggetti: {subs}   mappa: {args.kind}")

    def to_surface(vals364, fill=np.nan):
        img = np.full((H, W), fill, dtype=np.float32)
        ok = patch_of_pixel >= 0
        img[ok] = vals364[patch_of_pixel[ok]]
        img = np.nan_to_num(img, nan=0.0)
        surf = np.asarray(R.inverse(img[None]))       # [1, V] o [1, 58212]
        return surf.reshape(-1)

    out = {}
    for s in subs:
        key = f"{args.kind}_{s}"
        if key not in data:
            raise SystemExit(f"chiave {key} assente in {args.maps}")
        v = to_surface(np.asarray(data[key], np.float64))
        out[s] = v
        print(f"   {s}: superficie {v.shape}  "
              f"non nulli {int((v != 0).sum())}  "
              f"range [{v[v != 0].min():.3f}, {v[v != 0].max():.3f}]")

    grp = np.mean([out[s] for s in subs], axis=0)
    out["group"] = grp

    # ------------------------------------------------------- 4. round-trip
    print(f"\n=== 4. VERIFICA ROUND-TRIP ===")
    probe = np.arange(N_EXPECTED, dtype=np.float64)
    surf = to_surface(probe)
    back = np.asarray(R.transform(surf[None], interpolation="nearest")).reshape(H, W)
    ok = patch_of_pixel >= 0
    agree = (np.round(back[ok]) == patch_of_pixel[ok]).mean()
    print(f"   pixel che tornano al patch corretto: {100*agree:.2f}%")
    if agree < 0.95:
        print("   !! sotto il 95%: la corrispondenza non e' iniettiva come assunto.")
        print("   !! Probabilmente piu' pixel mappano sullo stesso vertice.")
    else:
        print("   catena coerente")

    fout = os.path.join(args.outdir, f"participation_fslr_{args.kind}.npz")
    np.savez_compressed(
        fout, patch_of_pixel=patch_of_pixel, valid_patches=valid,
        cover=cover, threshold=thr, point_mask=point_mask,
        **{f"surf_{k}": v for k, v in out.items()})
    print(f"\nsalvato in {fout}")
    print("Prossimo: controllo positivo (correlazione con motion -> aree visive),")
    print("poi confronto con Yeo e gradiente di Margulies via neuromaps (fslr 32k/164k).")


if __name__ == "__main__":
    main()
