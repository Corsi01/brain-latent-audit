#!/usr/bin/env python3
"""
f0_probe.py — sonde controllate attraverso CortexMAE-F.

Replica ESATTAMENTE il preprocessing di extract_latents.py:
  finestra raw (16, 77763) dalla cache 1 Hz
    -> z-score per pixel sui 16 frame  (axis=0)
    -> sample dict {bold fp16, mean, std, tr}
    -> model.run_embedding(...)  -> patch_embeds (1456, 768)

Conseguenza del preprocessing: l'input all'encoder ha media temporale ESATTAMENTE
zero e sd ESATTAMENTE uno per ogni pixel. Lo sviluppo
    m_s = E[f(X_s)] = f(0) + 1/2 (curvatura . Cov[X_s]) + ...
e' quindi centrato correttamente, e la diagonale di Cov e' identica per tutti i
soggetti: l'unica variazione residua e' nella struttura di correlazione spaziale.

Sonde (specificate nello spazio RAW, la normalizzazione avviene nel forward):

  zeros     finestra nulla                 -> f(0) puro. Bias di rete.
  gaussian  rumore i.i.d. per pixel        -> f(0) + curvatura con covarianza
                                              spaziale DIAGONALE. Zero biologia.
  shuffled  frame reali ricombinati        -> + covarianza spaziale reale e
                                              baseline per-frame, senza struttura
                                              temporale dello stimolo.
  real      finestre reali intatte         -> riferimento, e verifica di fedelta'
                                              contro latents_lag4.

Nota: una sonda "meanmap" non ha senso qui — dopo la z-score per clip la media
delle finestre e' identicamente zero, quindi coincide con `zeros`.

Uso:
    python f0_probe.py --out $CS_CACHE/f0/probes.npz --n-draws 64
    python f0_probe.py --out /tmp/t.npz --n-draws 2 --n-real 16 --check-fidelity
"""
import argparse
import glob
import os
import numpy as np
import torch

NUM_FRAMES = 16
FLAT_DIM = 77763


# ---------------------------------------------------------------------------
# Plug-in compilato da extract_latents.py — deve restare identico a quello.
# ---------------------------------------------------------------------------

def load_model(name="cortex_mae_flat", device=None):
    from cortex_mae import CortexMAE
    model = CortexMAE.from_pretrained(name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    print(f"      model={name}  device={device}")
    return model


def encode_window(model, bold_raw, batch_size=8):
    """
    bold_raw : np.ndarray [16, 77763] float32, spazio RAW (pre z-score di clip).
    Ritorna  : np.ndarray [1456, 768] float32.
    La normalizzazione qui dentro e' copiata verbatim da extract_latents.py.
    """
    bold_raw = np.ascontiguousarray(bold_raw, dtype=np.float32)
    assert bold_raw.shape == (NUM_FRAMES, FLAT_DIM), bold_raw.shape

    mu = bold_raw.mean(axis=0, keepdims=True)
    sd = bold_raw.std(axis=0, keepdims=True)
    bold = np.where(sd > 1e-6, (bold_raw - mu) / np.clip(sd, 1e-6, None), 0.0)

    sample = {
        "bold": torch.from_numpy(bold.astype(np.float16)),
        "mean": torch.from_numpy(mu.astype(np.float32).squeeze(0)),
        "std":  torch.from_numpy(sd.astype(np.float32).squeeze(0)),
        "tr":   1.0,
    }
    embeds = model.run_embedding(sample, tr=1.0, batch_size=batch_size)
    patch = embeds.patch_embeds.float().numpy()
    return (patch[0] if patch.shape[0] == 1 else patch).astype(np.float32)


# ---------------------------------------------------------------------------

def load_real_windows(flat_dir, n, rng):
    files = sorted(glob.glob(os.path.join(flat_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"nessun .npy in {flat_dir}")
    print(f"      {len(files)} run trovati")
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > 20 * n:
            raise RuntimeError("run piu' corti di 16 frame?")
        f = files[rng.integers(len(files))]
        arr = np.load(f, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != FLAT_DIM:
            raise ValueError(f"{os.path.basename(f)}: forma {arr.shape}")
        if arr.shape[0] < NUM_FRAMES:
            continue
        t0 = int(rng.integers(arr.shape[0] - NUM_FRAMES + 1))
        out.append(np.asarray(arr[t0:t0 + NUM_FRAMES], dtype=np.float32))
    return np.stack(out)


def check_fidelity(model, flat_dir, latents_dir, batch_size):
    """Ri-estrae una clip gia' presente in latents_lag4 e confronta.
    Se questo non passa, il plug-in non replica stage C e le sonde non valgono."""
    cands = sorted(glob.glob(os.path.join(latents_dir, "*.npz")))
    if not cands:
        print("      nessun latente di riferimento, salto")
        return
    z = np.load(cands[0], allow_pickle=True)
    sub = str(z["subject"]); run_id = str(z["run_id"])
    t0, t1 = [int(v) for v in np.asarray(z["fmri_window"]).ravel()[:2]]
    npy = os.path.join(flat_dir, f"{sub}_{run_id}.npy")
    if not os.path.exists(npy):
        print(f"      {os.path.basename(npy)} assente, salto")
        return
    bold = np.load(npy).astype(np.float32)[t0:t1]
    got = encode_window(model, bold, batch_size)
    ref = np.asarray(z["patch_embeds"], dtype=np.float32)
    d = float(np.abs(got - ref).max())
    c = float(np.dot(got.ravel() / np.linalg.norm(got.ravel()),
                     ref.ravel() / np.linalg.norm(ref.ravel())))
    print(f"      {os.path.basename(cands[0])}: max|diff| = {d:.3e}   cos = {c:.6f}")
    if c < 0.999:
        print("      !! il plug-in NON replica stage C. Fermati e verifica.")
    else:
        print("      fedelta' ok")


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    ap.add_argument("--flat-dir", default=os.path.join(cache, "fmri_flat_1hz"))
    ap.add_argument("--latents-dir", default=os.path.join(cache, "latents_lag4"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=64)
    ap.add_argument("--n-real", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model", default="cortex_mae_flat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-fidelity", action="store_true", default=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[1/5] finestre reali ({args.n_real})")
    real = load_real_windows(args.flat_dir, args.n_real, rng)      # [n,16,77763]
    sd_pix = real.reshape(-1, FLAT_DIM).std(axis=0)
    print(f"      sd media per pixel nello spazio raw: {sd_pix.mean():.4f}")
    print("      (irrilevante per il test: la z-score di clip la azzera comunque)")

    print("[2/5] modello")
    model = load_model(args.model, None)

    if args.check_fidelity:
        print("[3/5] verifica di fedelta' contro latents_lag4")
        check_fidelity(model, args.flat_dir, args.latents_dir, args.batch_size)
    else:
        print("[3/5] verifica di fedelta' saltata")

    def run(x):
        e = encode_window(model, x, args.batch_size)
        if e.shape != (1456, 768):
            print(f"      !! forma {e.shape}, attesa (1456, 768)")
        return e

    probes = {}
    print("[4/5] sonda deterministica: zeros")
    probes["zeros"] = run(np.zeros((NUM_FRAMES, FLAT_DIM), np.float32))
    print("      ok")

    print(f"[5/5] sonde stocastiche, {args.n_draws} estrazioni")
    acc_g, acc_s, acc_r = [], [], []
    pool = real.reshape(-1, FLAT_DIM)
    for k in range(args.n_draws):
        acc_g.append(run(rng.standard_normal((NUM_FRAMES, FLAT_DIM)).astype(np.float32)))
        acc_s.append(run(pool[rng.integers(pool.shape[0], size=NUM_FRAMES)]))
        acc_r.append(run(real[rng.integers(real.shape[0])]))
        if (k + 1) % 16 == 0:
            print(f"      {k+1}/{args.n_draws}")

    for name, acc in (("gaussian", acc_g), ("shuffled", acc_s), ("real", acc_r)):
        a = np.stack(acc)
        probes[name] = a.mean(0)
        probes[name + "_sem"] = a.std(0) / np.sqrt(len(a))

    np.savez_compressed(args.out, **probes)
    print(f"\nsalvato in {args.out}")
    print("sonde:", ", ".join(k for k in probes if not k.endswith("_sem")))
    print(f"\nOra: python f0_analysis.py --latents {args.latents_dir} "
          f"--probes {args.out} --ref zeros")


if __name__ == "__main__":
    main()