################################################################################
#                SNI Reproducibility - Complete Command Reference               #
################################################################################

# Activate environment
conda activate sni

################################################################################
# PART 1: SNI EXPERIMENTS
################################################################################

#-------------------------------------------------------------------------------
# 1.1 Main Experiments (MCAR/MAR @ 30%, Table 2-3)
#-------------------------------------------------------------------------------
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_main.csv \
    --outdir results_sni_main \
    --n-jobs -1

python scripts/aggregate_results.py \
    --results-root results_sni_main \
    --outdir results_sni_main/_summary

python scripts/make_latex_table.py \
    --summary-csv results_sni_main/_summary/summary_agg.csv \
    --outdir results_sni_main/_tables

#-------------------------------------------------------------------------------
# 1.2 Ablation Study (SNI vs NoPrior vs HardPrior, Table 4)
#-------------------------------------------------------------------------------
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_ablation.csv \
    --outdir results_sni_ablation \
    --n-jobs -1

python scripts/aggregate_results.py \
    --results-root results_sni_ablation \
    --outdir results_sni_ablation/_summary

python scripts/make_latex_table.py \
    --summary-csv results_sni_ablation/_summary/summary_agg.csv \
    --outdir results_sni_ablation/_tables

#-------------------------------------------------------------------------------
# 1.3 MNAR Robustness (SNI vs SNI-M, Table 5)
#-------------------------------------------------------------------------------
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_mnar.csv \
    --outdir results_sni_mnar \
    --n-jobs -1

python scripts/aggregate_results.py \
    --results-root results_sni_mnar \
    --outdir results_sni_mnar/_summary

python scripts/make_latex_table.py \
    --summary-csv results_sni_mnar/_summary/summary_agg.csv \
    --outdir results_sni_mnar/_tables


################################################################################
# PART 2: BASELINE EXPERIMENTS
################################################################################

#-------------------------------------------------------------------------------
# 2.1 Classic Baselines - Main Settings (MCAR/MAR @ 30%)
#-------------------------------------------------------------------------------
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_main.csv \
    --outdir results_baselines_main \
    --n-jobs -1 \
    --skip-existing \
    --default-use-gpu false

python scripts/aggregate_results.py \
    --results-root results_baselines_main \
    --outdir results_baselines_main/_summary

python scripts/make_latex_table.py \
    --summary-csv results_baselines_main/_summary/summary_agg.csv \
    --outdir results_baselines_main/_tables

#-------------------------------------------------------------------------------
# 2.2 Classic Baselines - MNAR Settings
#-------------------------------------------------------------------------------
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_mnar.csv \
    --outdir results_baselines_mnar \
    --n-jobs -1 \
    --skip-existing \
    --default-use-gpu false

python scripts/aggregate_results.py \
    --results-root results_baselines_mnar \
    --outdir results_baselines_mnar/_summary

python scripts/make_latex_table.py \
    --summary-csv results_baselines_mnar/_summary/summary_agg.csv \
    --outdir results_baselines_mnar/_tables

#-------------------------------------------------------------------------------
# 2.3 Deep Baselines (GAIN + MIWAE) - GPU recommended, run sequentially
#-------------------------------------------------------------------------------
# Note: Split into batches due to GPU memory constraints

# Batch 1: rows 0-100
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_deep.csv \
    --outdir results_baselines_deep \
    --n-jobs 1 \
    --row-start 0 --row-end 100 \
    --skip-existing \
    --default-use-gpu true

# Batch 2: rows 100-200
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_deep.csv \
    --outdir results_baselines_deep \
    --n-jobs 1 \
    --row-start 100 --row-end 200 \
    --skip-existing \
    --default-use-gpu true

# Batch 3: rows 200-end
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_baselines_deep.csv \
    --outdir results_baselines_deep \
    --n-jobs 1 \
    --row-start 200 --row-end -1 \
    --skip-existing \
    --default-use-gpu true

python scripts/aggregate_results.py \
    --results-root results_baselines_deep \
    --outdir results_baselines_deep/_summary

python scripts/make_latex_table.py \
    --summary-csv results_baselines_deep/_summary/summary_agg.csv \
    --outdir results_baselines_deep/_tables


################################################################################
# PART 3: SANITY CHECK - DEPENDENCY RECOVERY (Supplementary Section S5)
################################################################################

#-------------------------------------------------------------------------------
# 3.1 Generate Synthetic Data
#-------------------------------------------------------------------------------
mkdir -p data/synth_s5

# Linear Gaussian setting
for s in 2025 2026 2027 2028 2029; do
    python scripts/synth_generate_s5.py \
        --outdir data/synth_s5 \
        --setting linear_gaussian \
        --seed $s \
        --n 2000 --n-cont 10 --n-cat 2 \
        --mechanism MAR --missing-rate 0.30 \
        --driver-cols x0
done

# Nonlinear Mixed setting (product interactions)
for s in 2025 2026 2027 2028 2029; do
    python scripts/synth_generate_s5.py \
        --outdir data/synth_s5 \
        --setting nonlinear_mixed \
        --seed $s \
        --n 2000 --n-cont 10 --n-cat 2 \
        --mechanism MAR --missing-rate 0.30 \
        --driver-cols x0
done

#-------------------------------------------------------------------------------
# 3.2 Run Sanity Check Experiments
#-------------------------------------------------------------------------------
# Note: interaction_xor setting is auto-generated if missing

mkdir -p results_sanity_s5

python scripts/sanity_check_v2_s5.py \
    --data-dir data/synth_s5 \
    --outdir results_sanity_s5 \
    --settings linear_gaussian nonlinear_mixed interaction_xor \
    --seeds 2025 2026 2027 2028 2029 \
    --mechanism MAR --missing-rate 0.30 \
    --use-gpu true \
    --epochs 50 --num-heads 4 --emb-dim 32 --batch-size 128 --max-iters 2

# Outputs:
#   - results_sanity_s5/metrics_per_run.csv    : Per-seed metrics
#   - results_sanity_s5/table_S21.csv          : Aggregated mean±std
#   - results_sanity_s5/table_S21.tex          : LaTeX table
#   - results_sanity_s5/D_*.csv                : Learned dependency matrices


################################################################################
# PART 4: EXT1 - INTERPRETABILITY AUDIT & DOWNSTREAM VALIDATION
################################################################################

#-------------------------------------------------------------------------------
# 4.1 Exp1: Interpretability Audit Story (Leakage/Proxy Injection)
#-------------------------------------------------------------------------------
# Example: MIMIC, audit ALARM
python ext1/scripts/exp1_audit_story_leakage.py \
  --input-complete data/MIMIC_complete.csv \
  --dataset-name MIMIC \
  --categorical-vars SpO2 ALARM \
  --continuous-vars RESP ABP SBP DBP HR PULSE \
  --audit-target ALARM \
  --mechanism MAR --missing-rate 0.30 \
  --seed 2026 \
  --outdir results_ext1/audit_mimic_alarm \
  --run-without-leak true \
  --use-gpu false

# Key outputs:
#   results_ext1/audit_mimic_alarm/audit_report.md
#   results_ext1/audit_mimic_alarm/with_leak/dependency_matrix.csv
#   results_ext1/audit_mimic_alarm/with_leak/audit_top_sources.csv
#   results_ext1/audit_mimic_alarm/audit_comparison.csv

#-------------------------------------------------------------------------------
# 4.2 Exp2: Downstream Task Validation (Impute -> Predict)
#-------------------------------------------------------------------------------
# Example: NHANES, predict metabolic_score, fairness by gender_std
python ext1/scripts/exp2_downstream_task_validation.py \
  --input-complete data/NHANES_complete.csv \
  --dataset-name NHANES \
  --target-col metabolic_score \
  --categorical-cols gender_std age_band \
  --continuous-cols waist_circumference systolic_bp diastolic_bp triglycerides hdl_cholesterol fasting_glucose age bmi hba1c \
  --mechanism MAR --missing-rate 0.30 \
  --mar-driver-cols age gender_std \
  --fairness-col gender_std \
  --imputers SNI MissForest MeanMode MICE \
  --seeds 1 2 3 5 8 \
  --outdir results_ext1/downstream_nhanes \
  --sni-use-gpu false \
  --baseline-use-gpu false \
  --save-missing true \
  --save-imputed false

# Key outputs:
#   results_ext1/downstream_nhanes/metrics_per_seed.csv
#   results_ext1/downstream_nhanes/metrics_summary.csv


################################################################################
# PART 5: UTILITY COMMANDS
################################################################################

#-------------------------------------------------------------------------------
# 5.1 Resume Failed Runs
#-------------------------------------------------------------------------------
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_sni_main.csv \
    --outdir results_sni_main \
    --n-jobs -1 \
    --skip-existing

#-------------------------------------------------------------------------------
# 5.2 Check Progress
#-------------------------------------------------------------------------------
# Count completed experiments
find results_sni_main -name "metrics_summary.json" | wc -l

# Count failed experiments
find results_sni_main -name "error.log" | wc -l

# View error logs
find results_sni_main -name "error.log" -exec echo "=== {} ===" \; -exec cat {} \;

#-------------------------------------------------------------------------------
# 5.3 Verify Output Integrity
#-------------------------------------------------------------------------------
python - <<'PY'
import pandas as pd
import glob

paths = glob.glob("results_sni_main/*/imputed.csv")
print(f"Total imputed files: {len(paths)}")
if paths:
    df = pd.read_csv(paths[0])
    print(f"Sample: {paths[0]}")
    print(f"Any NaN: {df.isna().any().any()}")
PY

#-------------------------------------------------------------------------------
# 5.4 Resource Monitoring
#-------------------------------------------------------------------------------
# Monitor CPU
htop

# Monitor GPU
watch -n1 nvidia-smi

# Combined monitoring
watch -n2 'echo "=== CPU ===" && top -b -n1 | head -8 && echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'


################################################################################
# PART 6: OPTIONAL EXPERIMENTS (Extended Analysis)
################################################################################

#-------------------------------------------------------------------------------
# 6.1 Missing Rate Sweep (5%, 10%, 20%, 30%, 40%, 50%)
#-------------------------------------------------------------------------------
python scripts/run_manifest_parallel.py \
    --manifest data/manifest_options/manifest_sni_rate_sweep.csv \
    --outdir results_sni_rate_sweep \
    --n-jobs 8 \
    --skip-existing

python scripts/aggregate_results.py \
    --results-root results_sni_rate_sweep \
    --outdir results_sni_rate_sweep/_summary

#-------------------------------------------------------------------------------
# 6.2 Full Baseline Comparison (All Settings)
#-------------------------------------------------------------------------------
python scripts/run_manifest_baselines.py \
    --manifest data/manifest_options/manifest_baselines_main_all.csv \
    --outdir results_baselines_main_all \
    --n-jobs 8 \
    --skip-existing \
    --default-use-gpu false

python scripts/aggregate_results.py \
    --results-root results_baselines_main_all \
    --outdir results_baselines_main_all/_summary


################################################################################
# EXPERIMENT SUMMARY
################################################################################
#
# Experiment              | Manifest                        | Est. Time | n-jobs
# ------------------------|--------------------------------|-----------|--------
# SNI Main                | manifest_sni_main.csv          | 1-2h      | -1
# SNI Ablation            | manifest_sni_ablation.csv      | 2-3h      | -1
# SNI MNAR                | manifest_sni_mnar.csv          | 3-4h      | -1
# Baselines Main          | manifest_baselines_main.csv    | 4-6h      | -1
# Baselines MNAR          | manifest_baselines_mnar.csv    | 6-8h      | -1
# Baselines Deep (GPU)    | manifest_baselines_deep.csv    | 8-12h     | 1
# Sanity Check            | (synthetic)                    | 1-2h      | N/A
# Ext1 Audit Story        | (standalone script)            | <30min    | N/A
# Ext1 Downstream         | (standalone script)            | <1h       | N/A
#
# Note: Estimated times based on 32-core CPU + 1 GPU server.
#
################################################################################
