# Experiment Manifests (SNI paper)

Last updated: 2026-01-04

This repository uses **manifest CSV files** under `data/` to drive batch experiments.
Each row in a manifest corresponds to **one** run (dataset × missingness × rate × method/variant × seed).

## Runners

- **SNI / variants**: `python scripts/run_manifest_parallel.py --manifest <CSV> --outdir <DIR> --n-jobs <K>`
- **Baselines**: `python scripts/run_manifest_baselines.py --manifest <CSV> --outdir <DIR> --n-jobs <K>`

All paths inside manifest CSVs are **relative to the repo root**.

## Data layout (expected)

After preparing the data, the directory should contain:

- `data/*_complete.csv` (complete tables, no missing)
- `data/<DATASET>/<DATASET>_<MECH>_<RATE>.csv` (missing tables)
  - `<MECH>` in `{MCAR, MAR, MNAR}`
  - `<RATE>` in `{10per, 30per, 50per}`

## Seed policy (reproducibility)

All manifests use the same five seeds: `{1, 2, 3, 5, 8}`.

## Manifest catalog

### SNI (Statistical-Neural Interaction)

- `data/manifest_sni_main.csv`  (**60 runs**)  
  Main results: **SNI**, `MCAR+MAR`, `30per`, 6 datasets × 5 seeds.  
  Used for the main tables/figures in the paper.

- `data/manifest_sni_ablation.csv`  (**45 runs**)  
  Ablation on **MAR 30%** for `{MIMIC, NHANES, Concrete}` with **{SNI, NoPrior, HardPrior}** × 5 seeds.  
  Used to validate the necessity of adaptive priors.

- `data/manifest_sni_mnar.csv`  (**180 runs**)  
  MNAR robustness: `{SNI, SNI-M}` on **MNAR** at `{10per,30per,50per}` × 6 datasets × 5 seeds.

- `data/manifest_sni.csv`  (**420 runs, legacy/superset**)  
  A single “all-in-one” manifest that **covers everything above** (and more), kept for convenience/backward compatibility.  
  Prefer the split manifests (`*_main/ablation/mnar`) for controllable runs.

**Naming convention** (SNI):
- `MAIN_<DS>_<MECH>_<RATE>_SNI_s<SEED>`
- `ABLA_<DS>_MAR_30per_<VARIANT>_s<SEED>`
- `MNAR_<DS>_MNAR_<RATE>_<VARIANT>_s<SEED>`

### Baselines

- `data/manifest_baselines_main.csv`  (**360 runs**)  
  `{MeanMode, KNN, MICE, MissForest, GAIN, MIWAE}` on `MCAR+MAR`, `30per`, 6 datasets × 5 seeds.

- `data/manifest_baselines_mnar.csv`  (**540 runs**)  
  Same 6 baselines on **MNAR** at `{10per,30per,50per}`, 6 datasets × 5 seeds.

- `data/manifest_baselines_deep.csv`  (**300 runs**)  
  Deep baselines only: `{GAIN, MIWAE}` on:
  - `MCAR 30per` and `MAR 30per` (comparison with main tables)
  - `MNAR {10,30,50}per` (robustness)

## Known overlaps (why some settings appear in multiple manifests)

Some manifests intentionally **repeat the same experimental setting** so that each group can be executed and archived independently.

1) **SNI main vs ablation**  
- `manifest_sni_ablation.csv` includes **15** SNI runs that are also present in `manifest_sni_main.csv`  
  (datasets `{MIMIC, NHANES, Concrete}` × `MAR 30per` × 5 seeds).  
  **Reason**: make the ablation folder self-contained (SNI reference + ablated variants), so you can run/aggregate it without depending on `results_sni/main`.

2) **Baselines deep vs baselines main/mnar**  
- `manifest_baselines_deep.csv` is a **subset** of `manifest_baselines_main.csv ∪ manifest_baselines_mnar.csv` for `{GAIN, MIWAE}`.  
  **Reason**: deep baselines are GPU-friendly but do not parallelize well on a single GPU; keeping them separate lets you run them sequentially with `--default-use-gpu true` while running classical baselines in CPU-parallel mode.

3) **`manifest_sni.csv`**  
- This is a legacy **superset manifest**. It is expected to overlap with all split manifests.  
  **Reason**: one-shot runs and backward compatibility with earlier scripts.

## Recommended run plan (paper reproduction)

### SNI
```bash
# Main results (MCAR/MAR 30%)
python scripts/run_manifest_parallel.py \
  --manifest data/manifest_sni_main.csv \
  --outdir results_sni/main \
  --n-jobs 8 --skip-existing

# Ablation (MAR 30%)
python scripts/run_manifest_parallel.py \
  --manifest data/manifest_sni_ablation.csv \
  --outdir results_sni/ablation \
  --n-jobs 8 --skip-existing

# MNAR robustness (SNI vs SNI-M)
python scripts/run_manifest_parallel.py \
  --manifest data/manifest_sni_mnar.csv \
  --outdir results_sni/mnar \
  --n-jobs 8 --skip-existing
```

### Baselines (classical)
```bash
# Main baselines (MCAR/MAR 30%)
python scripts/run_manifest_baselines.py \
  --manifest data/manifest_baselines_main.csv \
  --outdir results_baselines/main \
  --n-jobs 8 --skip-existing \
  --default-use-gpu false

# MNAR baselines (10/30/50%)
python scripts/run_manifest_baselines.py \
  --manifest data/manifest_baselines_mnar.csv \
  --outdir results_baselines/mnar \
  --n-jobs 8 --skip-existing \
  --default-use-gpu false
```

### Baselines (deep, recommended GPU schedule)
```bash
# Safest: single worker on the single GPU
python scripts/run_manifest_baselines.py \
  --manifest data/manifest_baselines_deep.csv \
  --outdir results_baselines/deep \
  --n-jobs 1 --skip-existing \
  --default-use-gpu true

# Optional: slice into chunks to make reruns controllable
python scripts/run_manifest_baselines.py \
  --manifest data/manifest_baselines_deep.csv \
  --outdir results_baselines/deep \
  --n-jobs 1 --row-start 0 --row-end 100 --skip-existing \
  --default-use-gpu true
```

## Aggregation / reporting

Aggregate each results root separately to avoid double-counting overlaps:
```bash
python scripts/aggregate_results.py --results-root results_sni/main --outdir results_sni/main/_summary
python scripts/aggregate_results.py --results-root results_sni/ablation --outdir results_sni/ablation/_summary
python scripts/aggregate_results.py --results-root results_sni/mnar --outdir results_sni/mnar/_summary

python scripts/aggregate_results.py --results-root results_baselines/main --outdir results_baselines/main/_summary
python scripts/aggregate_results.py --results-root results_baselines/mnar --outdir results_baselines/mnar/_summary
python scripts/aggregate_results.py --results-root results_baselines/deep --outdir results_baselines/deep/_summary
```

## Optional improvements (deduplicated & extra experiments)

These are **optional** files to make future reruns cleaner and/or to strengthen the paper:

### (A) Deduplicated baseline manifests (recommended for reruns)
- `data/manifest_baselines_main_classic.csv` (240 runs): `MeanMode/KNN/MICE/MissForest` only (no deep).
- `data/manifest_baselines_mnar_classic.csv` (360 runs): same (no deep).
Run deep baselines separately with `manifest_baselines_deep.csv` to avoid duplicates and GPU contention.

### (B) Extra: MCAR/MAR missing-rate sweep (10% & 50%)
If the paper/supplement would benefit from an explicit sensitivity-to-missing-rate plot under MCAR/MAR, run:
- `data/manifest_sni_rate_sweep.csv` (120 runs): SNI on `MCAR+MAR` at `{10per,50per}`.
- `data/manifest_baselines_rate_sweep_missforest.csv` (120 runs): MissForest on `MCAR+MAR` at `{10per,50per}`.

This keeps the added workload small while answering a common reviewer question: **how does performance degrade with missing rate** under standard mechanisms.

## Notes for clean experiment bookkeeping

1. Archive the exact manifest CSV(s) together with the results folder.
2. Keep a copy of the console log:
   ```bash
   python scripts/run_manifest_parallel.py ... |& tee results_sni/main/run.log
   ```
3. Record environment versions (`python -V`, `pip freeze`, CUDA driver) in the results folder.
