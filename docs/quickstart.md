# Quick Start

## 1. Smoke Test (verify installation)

```bash
python scripts/paper_quickstart.py --smoke-test --outdir results_quick
```

This runs a comprehensive smoke test (~5 min on CPU):
- Concrete (pure continuous) + MIMIC (mixed-type, if available) SNI experiments
- MeanMode, KNN, MICE baselines
- v0.3 artifact checks (dependency_matrix, lambda_per_head, etc.)
- Full post-process chain: aggregate -> merge -> LaTeX -> figures

## 2. Run ALL Paper Experiments (one command)

```bash
python scripts/run_all_experiments.py \
    --outdir results_all \
    --skip-existing \
    --n-jobs-cpu 8 \
    --n-jobs-gpu 1
```

This runs all 7 stages in sequence:
- Stage 1: SNI experiments (main, ablation, MNAR, rate sweep)
- Stage 2: Classic baselines (MCAR/MAR + MNAR)
- Stage 3: Deep baselines (GAIN, MIWAE, HyperImpute, TabCSDI)
- Stage 4: Sanity check (synthetic dependency recovery)
- Stage 5: Ext1 (interpretability audit, downstream validation)
- Stage 6: Ext2 (per-class, SHAP, significance, Impute-Predict)
- Stage 7: Aggregate + merge + LaTeX tables + figures

### Run Selected Stages

```bash
# SNI + classic baselines + postprocess only
python scripts/run_all_experiments.py --outdir results_all --stages 1 2 7 --skip-existing

# Re-generate tables and figures from existing results
python scripts/run_all_experiments.py --outdir results_all --stages 7

# Preview all commands without executing
python scripts/run_all_experiments.py --outdir results_all --dry-run
```

## 3. Post-Process Only (from existing results)

```bash
python scripts/paper_quickstart.py --postprocess-only \
  --sni-results results_sni_v03_main \
  --baseline-results results_baselines \
  --outdir results_paper
```

This merges existing SNI and baseline results, then generates tables and figures.

## Output Structure

```
results_all/
├── sni_v03_main/              # Stage 1: SNI main experiments
├── sni_v03_ablation_lambda/   # Stage 1: Lambda ablation
├── sni_mnar/                  # Stage 1: MNAR robustness
├── sni_rate_sweep/            # Stage 1: Rate sweep
├── baselines_main/            # Stage 2: Classic baselines (MCAR/MAR)
├── baselines_mnar/            # Stage 2: Classic baselines (MNAR)
├── baselines_deep/            # Stage 3: Deep baselines (GAIN, MIWAE)
├── baselines_new/             # Stage 3: New baselines (HyperImpute, TabCSDI)
├── sanity_s5/                 # Stage 4: Synthetic dependency recovery
├── ext1/                      # Stage 5: Interpretability + downstream
├── ext2/                      # Stage 6: Per-class, SHAP, significance
├── agg_sni_v03_main/          # Stage 7a: Aggregated summaries (one per group)
├── agg_baselines_main/        #   also supports summary_all_runs.csv fallback
├── agg_baselines_deep/        #
├── ...                        #
├── merged/                    # Stage 7b: Merged summary across all groups
│   └── merged_summary_agg.csv
├── tables/                    # Stage 7c: LaTeX tables (main profile)
├── tables_supplement/         # Stage 7d: LaTeX tables (supplement profile)
├── figures/                   # Stage 7e: Figures (main profile)
└── figures_supplement/        # Stage 7f: Figures (supplement profile)
```
