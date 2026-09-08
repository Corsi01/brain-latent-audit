# extra/

Secondary validation scripts — real controls, not part of the main argument.

- `retrieval_variants.py` — three alternative representations (unbinned voxels, CLS token, channel-PCA) probed to understand what each form of compression buys or costs. Informative, but superseded as a headline result once `pca_sweep.py`/`pca_loso.py` picked the operating point actually reported.
- `pca_sweep.py` — sweeps PCA rank k to find the smallest one that preserves both retrieval accuracy and the drive gradient. Exploratory: it justifies the k used elsewhere rather than being a result on its own.
- `pca_loso.py` — leave-two-subjects-out re-fit of that PCA basis, to check the sweep result wasn't fit on the same subjects it was evaluated on. A robustness check on `pca_sweep.py`, not an independent finding.
- `probe_grain_and_mean.py` — two diagnostic tests (split-half stability of the per-subject mean; within-category "grain" of the latent) that informed later analyses but are not part of the primary reported numbers.

Everything here runs and is referenced by the report; it's separated so the
top level of `validation/` holds only the five scripts that produce the
numbers in the main table.
