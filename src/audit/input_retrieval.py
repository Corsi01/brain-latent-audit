#!/usr/bin/env python3
"""
input_retrieval.py — retrieval inter-soggetto direttamente sulle FLAT MAP DI INPUT.

Domanda: quanta della capacita' di identificazione e' gia' nello spazio di input,
prima di qualunque encoder?

Il latente e' una funzione deterministica dell'input, quindi per data processing
inequality non puo' contenere informazione che l'input non ha. Se l'input da'
gia' ~26x il caso, nessuna rete aggiunge informazione e la discordanza tra le
varianti permuted/fresh sul retrieval e' irrilevante: e' tutta struttura
ereditata. Se l'input da' molto meno, l'architettura contribuisce e la
discrepanza va capita.

Usa esattamente le stesse 622 coppie (soggetto, clip) e la stessa finestra
temporale dei latenti, cosi' il confronto e' appaiato per costruzione.

Due spazi di input:
  raw        valori della cache 1 Hz (gia' z-scorati per pixel sul run)
  zscored    dopo la z-score di clip di extract_latents.py — cioe' ESATTAMENTE
             il tensore che entra nell'encoder

Due trasformazioni, per ciascuno:
  nessuna
  centering A per soggetto: media sui campioni per ogni (frame, pixel)
             (l'analogo del centering B non esiste nello spazio di input: li'
              non c'e' un asse canale da cui estrarre un vettore condiviso)

RAM: ~3.1 GB per lo spazio di input (622 x 16 x 77763 float32). Gira su nodo
di calcolo; sul login node puo' essere stretto.

Uso:
    python input_retrieval.py --latents $CS_CACHE/latents_lag4
"""
import argparse
import glob
import itertools
import os
import gc

import numpy as np

NUM_FRAMES = 16
FLAT_DIM = 77763


def retrieval(X, subj, clip):
    """X: [N, F] float32. Ritorna (top1, pct_rank, n_candidati)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    t1, pr, nn = [], [], 0
    for a, b in itertools.permutations(sorted(set(subj)), 2):
        ia, ib = np.where(subj == a)[0], np.where(subj == b)[0]
        common = np.intersect1d(clip[ia], clip[ib])
        if len(common) < 5:
            continue
        ia = ia[np.isin(clip[ia], common)]
        ia = ia[np.argsort(clip[ia])]
        ib = ib[np.isin(clip[ib], common)]
        ib = ib[np.argsort(clip[ib])]
        S = Xn[ia] @ Xn[ib].T
        n = S.shape[0]
        nn = n
        rank = np.argmax(np.argsort(-S, axis=1) == np.arange(n)[:, None], axis=1)
        t1.append((rank == 0).mean())
        pr.append((rank / (n - 1)).mean())
    return float(np.mean(t1)), float(np.mean(pr)), nn


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    ap.add_argument("--latents", default=os.path.join(cache, "latents_lag4"))
    ap.add_argument("--flat-dir", default=os.path.join(cache, "fmri_flat_1hz"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []
    def P(s=""):
        print(s, flush=True)
        lines.append(s)

    refs = sorted(glob.glob(os.path.join(args.latents, "*.npz")))
    if not refs:
        raise SystemExit(f"nessun latente in {args.latents}")
    P(f"{len(refs)} coppie (soggetto, clip) da {os.path.basename(args.latents)}")

    # --- carica le finestre di input nello stesso ordine dei latenti ---------
    N = len(refs)
    X = np.empty((N, NUM_FRAMES, FLAT_DIM), dtype=np.float32)
    subj = np.empty(N, dtype=object)
    clip = np.empty(N, dtype=object)
    run_cache, cur_key = {}, None
    for i, f in enumerate(refs):
        z = np.load(f, allow_pickle=True)
        sub, run_id = str(z["subject"]), str(z["run_id"])
        subj[i], clip[i] = sub, str(z["clip_id"])
        t0, t1_ = [int(v) for v in np.asarray(z["fmri_window"]).ravel()[:2]]
        key = (sub, run_id)
        if key != cur_key:
            run_cache = {key: np.load(
                os.path.join(args.flat_dir, f"{sub}_{run_id}.npy")).astype(np.float32)}
            cur_key = key
        X[i] = run_cache[key][t0:t1_]
        if (i + 1) % 200 == 0:
            P(f"   caricate {i+1}/{N}")
    del run_cache
    gc.collect()
    subj, clip = subj.astype(str), clip.astype(str)
    P(f"   input: {X.shape}  ({X.nbytes/1e9:.2f} GB)")

    results = []

    def evaluate(tag, A):
        subs = sorted(set(subj))
        flat = A.reshape(N, -1)
        t1, pr, n = retrieval(flat, subj, clip)
        P(f"   {tag:34s} top-1 {t1:.4f} ({t1*n:5.1f}x)  pct {pr:.4f}")
        results.append((tag, t1, pr, n))
        # centering A per soggetto: media sui campioni, per (frame, pixel)
        B = A.copy()
        for s in subs:
            m = B[subj == s].mean(axis=0)
            B[subj == s] -= m
        t1c, prc, _ = retrieval(B.reshape(N, -1), subj, clip)
        gain = (t1c / t1 - 1) * 100 if t1 > 0 else float("nan")
        P(f"   {tag + ' + centering A':34s} top-1 {t1c:.4f} ({t1c*n:5.1f}x)  "
          f"pct {prc:.4f}   [{gain:+.1f}%]")
        results.append((tag + " + centering A", t1c, prc, n))
        del B
        gc.collect()

    P("\n=== SPAZIO DI INPUT: cache 1 Hz (z-score per pixel sul run) ===")
    evaluate("raw", X)

    P("\n=== SPAZIO DI INPUT: dopo z-score di clip (cio' che vede l'encoder) ===")
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    Xz = np.where(sd > 1e-6, (X - mu) / np.clip(sd, 1e-6, None), 0.0).astype(np.float32)
    del X, mu, sd
    gc.collect()
    evaluate("z-scored", Xz)

    P("\n=== Lettura ===")
    P("   Confronta con: latente addestrato raw 0.1715 (26.6x), centrato 0.2409 (37.3x).")
    P("   Se l'input z-scorato e' gia' vicino a 26x, il retrieval e' informazione")
    P("   dei dati e nessuna rete la crea. Se e' molto sotto, l'encoder contribuisce")
    P("   e la discordanza permuted/fresh va spiegata.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        print(f"\nreport in {args.out}")


if __name__ == "__main__":
    main()
