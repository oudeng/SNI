# Results Directory Structure

## Per-Experiment Output

Each experiment (SNI or baseline) produces a directory with the following structure:

```
{exp_id}/
├── imputed.csv                 # Imputed dataset
├── metrics_summary.json        # Summary metrics (JSON)
├── metrics_summary.csv         # Summary metrics (CSV)
├── metrics_per_feature.csv     # Per-feature metrics
├── run_config.json             # Run configuration snapshot
├── dependency_matrix.csv       # Feature dependency matrix (SNI only)
├── dependency_network_edges.csv # Network edges (SNI only)
├── dependency_matrix.png       # Dependency heatmap (SNI only)
├── convergence_curve.csv       # EM convergence (SNI v0.3)
├── lambda_per_head.csv         # Lambda values per head (SNI v0.3)
├── lambda_values.json          # Lambda values (SNI v0.3)
├── attention_maps/             # Per-target attention maps
│   └── {feature_name}.csv
├── lambda_traces/              # Per-target lambda traces
│   └── {feature_name}.csv
├── run.log                     # Stdout/stderr log (baselines)
└── error.log                   # Error log (if failed)
```

## Aggregated Output

```
{outdir}/
├── summary_all_runs.csv        # All experiments summary
├── parallel_run_meta.json      # Parallel run metadata
└── environment_snapshot.txt    # Environment info
```

## Full Pipeline Output (`run_all_experiments.py`)

When using `run_all_experiments.py --outdir results_all`, the following structure is produced:

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

## Post-Processing Output (standalone)

When using `paper_quickstart.py --postprocess-only`, results go to a separate directory:

```
results_paper/
├── environment_snapshot.txt
├── merged_summary_agg.csv
├── tables/
│   ├── Table_I_avg_rank.tex
│   ├── Table_II_MIMIC.tex
│   └── ...
└── figures/
    ├── Fig3_compact_main.png
    ├── Fig4_rate_sweep.png
    └── ...
```

## Naming Conventions

- **Experiment IDs**: `{dataset}_{mechanism}_{rate}p_{variant}_s{seed}` (e.g., `MIMIC_MCAR_30p_SNI_s42`)
- **Output prefixes**: `main_` for main paper, `supp_` for supplement, `all_` for full results
- **Runtime key**: Always `runtime_sec` (seconds as float)
