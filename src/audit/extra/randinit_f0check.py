#!/usr/bin/env python3
"""
randinit_f0check.py — la media della rete casuale E' f(0)?

Chiude il cerchio sull'ipotesi che avevamo messo alla prova:

    m_s = E_t[f(X_s(t))] ~= f(0) + curvatura ...

Sull'encoder addestrato cos(m_B, f(0)) = 0.17, cioe' f(0) spiega il 3% della
varianza: ipotesi falsificata. Qui verifichiamo la stessa quantita' sulle reti
NON addestrate. Se li' e' ~1, il risultato e' particolarmente pulito da
riportare: l'ipotesi f(0) e' vera per le reti casuali e falsa per quella
addestrata, quindi la direzione media appresa e' un oggetto di natura diversa
da un bias architetturale.

Usa le stesse funzioni di randomizzazione di randinit_extract.py, quindi con lo
stesso seed i pesi sono identici a quelli che hanno generato i latenti.

Uso:
    python randinit_f0check.py --variant permuted --seed 0
    python randinit_f0check.py --variant permuted --seed 1
    python randinit_f0check.py --variant fresh    --seed 0
    python randinit_f0check.py --variant trained          # controllo
"""
import argparse
import glob
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))  # audit/ — randinit_extract.py lives there
from randinit_extract import (
    encode_window, get_encoder, randomize_permuted, randomize_fresh,
    NUM_FRAMES, FLAT_DIM,
)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(np.ravel(a)), unit(np.ravel(b))))


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    ap.add_argument("--variant", choices=["permuted", "fresh", "trained"],
                    default="permuted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latents-dir", default=None,
                    help="default: latents_lag4[_{variant}{seed}]")
    ap.add_argument("--model", default="cortex_mae_flat")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    lat_dir = args.latents_dir or (
        os.path.join(cache, "latents_lag4") if args.variant == "trained"
        else os.path.join(cache, f"latents_lag4_{args.variant}{args.seed}"))

    files = sorted(glob.glob(os.path.join(lat_dir, "*.npz")))
    if not files:
        raise SystemExit(f"nessun latente in {lat_dir}")
    print(f"latenti: {len(files)} da {os.path.basename(lat_dir)}")

    # --- m_B per soggetto ---------------------------------------------------
    acc, cnt = {}, {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        s = str(z["subject"])
        e = np.asarray(z["patch_embeds"], np.float32).mean(axis=0)   # [768]
        acc[s] = acc.get(s, 0.0) + e
        cnt[s] = cnt.get(s, 0) + 1
    mB = {s: acc[s] / cnt[s] for s in acc}
    subs = sorted(mB)

    # --- f(0) con gli stessi pesi -------------------------------------------
    from cortex_mae import CortexMAE
    model = CortexMAE.from_pretrained(args.model)
    model.set_device("cuda" if torch.cuda.is_available() else "cpu")
    if args.variant == "permuted":
        print("  ", randomize_permuted(get_encoder(model), args.seed))
    elif args.variant == "fresh":
        print("  ", randomize_fresh(get_encoder(model), args.seed))
    else:
        print("   pesi addestrati (nessuna randomizzazione)")

    z0 = encode_window(model, np.zeros((NUM_FRAMES, FLAT_DIM), np.float32),
                       args.batch_size).mean(axis=0)                  # [768]

    print(f"\n=== variante: {args.variant}  seed={args.seed} ===")
    print(f"   ||f(0)|| = {np.linalg.norm(z0):.3f}")
    cc = [cos(mB[s], z0) for s in subs]
    print(f"   cos(m_B, f(0)) = {min(cc):.4f} - {max(cc):.4f}")
    print(f"   varianza di m_B spiegata da f(0) = "
          f"{min(c**2 for c in cc):.3f} - {max(c**2 for c in cc):.3f}")
    cs = [cos(mB[a], mB[b]) for a, b in itertools.combinations(subs, 2)]
    print(f"   coseno inter-soggetto di m_B = {min(cs):.4f} - {max(cs):.4f}")
    print("\n   ~1 -> la media E' il bias architetturale f(0)")
    print("   bassa -> la media dipende dai dati, non dall'architettura")


if __name__ == "__main__":
    main()
