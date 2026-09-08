#!/usr/bin/env python3
"""
verify_patch_order.py — l'ordine dei 364 token e' davvero row-major?

models_mae.py riga 204 conferma il criterio di validita' (patch_num_obs > 0,
esattamente i 364 patch che toccano corteccia), ma le righe 222-228 mostrano un
`trim_patch_mask(..., shuffle=True)` seguito da `gather`. Se quel ramo fosse
attivo anche in forward_embedding, l'ordine dei token non sarebbe row-major e
qualunque mappa proiettata sulla superficie risulterebbe permutata.

Leggere il codice non basta. Questo script lo stabilisce per misura: perturba un
patch alla volta nell'input e osserva quale token risponde.

Verifica due cose insieme:
  1. la permutazione spaziale patch -> token
  2. il reshape 1456 -> (4, 364) time-major contro (364, 4) space-major

La perturbazione deve VARIARE NEL TEMPO: encode_window applica una z-score per
pixel sui 16 frame, quindi un pixel costante avrebbe sd = 0 e verrebbe azzerato.

Uso:
    python verify_patch_order.py                 # 364 patch, ~90 s
    python verify_patch_order.py --n-patches 24  # prova rapida
"""
import argparse
import os

import numpy as np
import torch

from randinit_extract import encode_window, NUM_FRAMES, FLAT_DIM

H, W = 224, 560
PATCH = 16
GH, GW = H // PATCH, W // PATCH          # 14 x 35
N_SP = 364
N_T = 4


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    ap.add_argument("--model", default="cortex_mae_flat")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-patches", type=int, default=N_SP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(cache, "f0", "surface"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from cortex_mae import CortexMAE, transforms
    R = transforms._FLAT_RESAMPLER
    mask = np.asarray(R.mask_)                       # [224, 560] bool
    pix_idx = np.flatnonzero(mask.ravel())           # 77763 posizioni nella griglia

    # patch validi in ordine row-major
    m = mask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3)
    m = m.reshape(GH * GW, PATCH * PATCH)
    valid = m.any(axis=1)
    assert valid.sum() == N_SP, f"{valid.sum()} patch validi, attesi {N_SP}"
    valid_ids = np.flatnonzero(valid)

    # per ogni patch valido: indici nel vettore a 77763
    grid_pos = np.full(H * W, -1, np.int64)
    grid_pos[pix_idx] = np.arange(len(pix_idx))
    patch_pixels = []
    for p in valid_ids:
        gr, gc = divmod(p, GW)
        sl = np.zeros((H, W), bool)
        sl[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH] = True
        sel = grid_pos[np.flatnonzero(sl.ravel() & mask.ravel())]
        patch_pixels.append(sel)
    print(f"{len(patch_pixels)} patch validi, "
          f"pixel per patch: min {min(len(s) for s in patch_pixels)} "
          f"max {max(len(s) for s in patch_pixels)}")

    model = CortexMAE.from_pretrained(args.model)
    model.set_device("cuda" if torch.cuda.is_available() else "cpu")

    base = encode_window(model, np.zeros((NUM_FRAMES, FLAT_DIM), np.float32),
                         args.batch_size)                       # [1456, 768]
    print(f"baseline: {base.shape}\n")

    n = min(args.n_patches, N_SP)
    order = np.linspace(0, N_SP - 1, n).astype(int) if n < N_SP else np.arange(N_SP)

    resp = np.zeros((n, base.shape[0]), np.float32)
    for i, p in enumerate(order):
        x = np.zeros((NUM_FRAMES, FLAT_DIM), np.float32)
        x[:, patch_pixels[p]] = rng.standard_normal(
            (NUM_FRAMES, len(patch_pixels[p]))).astype(np.float32) * 5.0
        e = encode_window(model, x, args.batch_size)
        resp[i] = np.linalg.norm(e - base, axis=1)
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{n}", flush=True)

    # ------------------------------------------------------------ diagnosi
    print("\n=== 1. TIME-MAJOR O SPACE-MAJOR ===")
    tm = resp.reshape(n, N_T, N_SP)          # ipotesi (4, 364)
    sm = resp.reshape(n, N_SP, N_T)          # ipotesi (364, 4)
    # sotto l'ipotesi giusta, la risposta si concentra su UNA posizione spaziale
    # replicata su tutti i blocchi temporali
    conc_tm = (tm.max(axis=2) / (tm.sum(axis=2) + 1e-9)).mean()
    conc_sm = (sm.max(axis=1) / (sm.sum(axis=1) + 1e-9)).mean()
    print(f"   concentrazione spaziale, ipotesi time-major : {conc_tm:.4f}")
    print(f"   concentrazione spaziale, ipotesi space-major: {conc_sm:.4f}")
    time_major = conc_tm >= conc_sm
    print(f"   -> {'TIME-MAJOR (4, 364)' if time_major else 'SPACE-MAJOR (364, 4)'}")

    A = tm if time_major else sm.transpose(0, 2, 1)      # [n, 4, 364]

    print("\n=== 2. PERMUTAZIONE SPAZIALE ===")
    sp = A.mean(axis=1)                                  # [n, 364]
    hit = sp.argmax(axis=1)
    identity = (hit == order).mean()
    print(f"   token di massima risposta = indice row-major: {100*identity:.1f}%")
    conc = (sp.max(axis=1) / (sp.sum(axis=1) + 1e-9))
    print(f"   selettivita' media: {conc.mean():.4f} "
          f"(1/364 = {1/N_SP:.4f} se nessuna localizzazione)")
    if identity > 0.95:
        print("   -> ordine ROW-MAJOR confermato. La mappa proiettata e' corretta.")
    elif conc.mean() < 0.05:
        print("   -> risposta NON localizzata: l'encoder mescola i patch,")
        print("      oppure la perturbazione e' troppo debole. Aumenta l'ampiezza.")
    else:
        print("   -> risposta localizzata ma PERMUTATA. Salvo la permutazione:")
        print("      va applicata alla mappa prima di proiettarla sulla superficie.")
        print(f"      primi 10: patch {list(order[:10])} -> token {list(hit[:10])}")

    print("\n=== 3. ORDINE TEMPORALE ===")
    p0 = order[len(order) // 2]
    x = np.zeros((NUM_FRAMES, FLAT_DIM), np.float32)
    x[0:4, patch_pixels[p0]] = rng.standard_normal(
        (4, len(patch_pixels[p0]))).astype(np.float32) * 5.0
    e = encode_window(model, x, args.batch_size)
    r = np.linalg.norm(e - base, axis=1)
    B = r.reshape(N_T, N_SP) if time_major else r.reshape(N_SP, N_T).T
    prof = B[:, hit[len(order) // 2]] if identity <= 0.95 else B[:, p0]
    prof = prof / (prof.sum() + 1e-9)
    print(f"   perturbati solo i frame 0-3; risposta per blocco temporale:")
    print("   " + "  ".join(f"t{k}={v:.3f}" for k, v in enumerate(prof)))
    print("   se il blocco 0 domina, i 4 blocchi sono in ordine cronologico")

    out = os.path.join(args.outdir, "patch_order.npz")
    np.savez_compressed(out, resp=resp, order=order, hit=hit,
                        time_major=time_major, identity=identity)
    print(f"\nsalvato in {out}")


if __name__ == "__main__":
    main()
