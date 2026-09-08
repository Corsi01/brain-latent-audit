# extra/

Secondary audit scripts — supporting checks, not the main chain of evidence.

- `randinit_f0check.py` — checks whether a random-weight encoder's mean output coincides with its f(0) bias term. Closes a side-question opened by `f0_analysis.py`/`randinit_compare.py`; not needed to follow the main argument.
- `map_stability.py` — checks whether the "participation" map is really direction-alignment or just local amplitude in disguise. A sanity check that motivates using cosine in the main maps, rather than a result reported on its own.
- `patch_to_surface.py` — the patch-to-cortex-vertex projection utility. Infrastructure used to render maps, not an analysis in itself.
- `atlas_comparison.py` — tests participation against the principal connectivity gradient. Superseded as the primary network test by `yeo_test.py` (a discrete DMN-vs-sensory contrast, more interpretable and less confounded); kept because it's reported alongside it in the audit.
- `plot_maps.py` — generic renderer: takes a 364-value map and produces a PNG on the flat map. A plotting utility, not an analysis.

Everything here runs and is referenced by the report; it's separated so the
top level of `audit/` holds the twelve scripts that carry the audit's main
chain of reasoning end to end.
