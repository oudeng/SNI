# SNI Repository Review Report

Date: 2026-02-15

## Phase 1: Structure Consistency

### Task 1.1: File Tree vs README Description

| Check | Status |
|-------|--------|
| `SNI_v0_3/` listed | PASS |
| `baselines/` listed (8 methods) | PASS |
| `scripts/` listed | PASS |
| `ext1/` listed | PASS |
| `ext2/` listed | PASS |
| `data/` listed | PASS |
| `configs/` listed | PASS |
| `docs/` listed | PASS |
| `utility_missing_data_gen_v1/` listed | FIXED (was missing) |

**Fix applied:** Added `utility_missing_data_gen_v1/` to README "Repository Structure" section.

**Undocumented scripts (low priority):** 4 utility scripts not mentioned in root docs:
- `scripts/clean_release.py` — release prep utility
- `scripts/merge_summaries.py` — used internally by `run_all_experiments.py` Stage 7b
- `scripts/reporting_constants.py` — shared constants (used by other scripts)
- `scripts/split_manifest_runner.py` — batch script generator

These are internal helpers and do not need root-level documentation.

### Task 1.2: Configs & Manifest Validation

| Check | Status |
|-------|--------|
| `configs/datasets.yaml` — 6 datasets with correct vars | PASS |
| `configs/paper_profiles.yaml` — main/supplement/all profiles | PASS |
| All manifests referenced in readme_CL.txt exist | PASS |
| 3 extra manifests in `manifest_options/` undocumented | ACCEPTABLE (optional variants) |

Manifest counts (header excluded):
- `manifest_sni_v03_main.csv`: 60 rows
- `manifest_sni_v03_ablation_lambda.csv`: 50 rows
- `manifest_baselines_main.csv`: 240 rows
- `manifest_baselines_deep.csv`: 300 rows
- `manifest_baselines_new.csv`: 120 rows

All `requirements*.txt` files exist and are correctly referenced.

---

## Phase 2: Experiment Completeness

### Task 2.1: Stage Coverage

| Stage | Description | README | readme_CL.txt |
|-------|-------------|--------|---------------|
| 1a | SNI v0.3 Main | PASS | PASS |
| 1b | SNI v0.3 Lambda Ablation | PASS | PASS |
| 1c | SNI MNAR | PASS | PASS |
| 1d | SNI Rate Sweep | PASS | PASS |
| 2a | Baselines Main (MCAR/MAR) | PASS | PASS |
| 2b | Baselines MNAR | PASS | PASS |
| 3a | Baselines Deep (GAIN/MIWAE) | PASS | PASS |
| 3b | Baselines New (HyperImpute/TabCSDI) | PASS | PASS |
| 4a | Sanity Check (synth S5) | PASS | PASS |
| 5a | Ext1 Audit ALARM | PASS | PASS |
| 5b | Ext1 Audit SBP | PASS | PASS |
| 5c | Ext1 Downstream NHANES | PASS | PASS |
| 6a | Ext2 Per-class Categorical | PASS | PASS |
| 6b | Ext2 SHAP Comparison | PASS | PASS |
| 6c | Ext2 Attention Rollout | PASS | PASS |
| 6d | Ext2 D Stability | PASS | PASS |
| 6e | Ext2 Significance Tests | PASS | PASS |
| 6f | Ext2 MIMIC Impute-Predict | PASS | PASS |
| 7a | Aggregation | PASS | PASS |
| 7b | Merge summaries | PASS | PASS |
| 7c-d | LaTeX tables (main/supplement) | PASS | PASS |
| 7e-f | Figures (main/supplement) | PASS | PASS |

All 22 sub-stages are documented with both design explanation and complete command lines.

### Task 2.2: Paper Table/Figure Mapping

`docs/evidence_map.md` maps all main paper artifacts (Tables I-V, Figs 3-6) and
supplementary materials (Tables S7-S9, Table VI) to their generating scripts. PASS.

---

## Phase 3: Command-Line Executability

### Task 3.1: readme_CL.txt Argument Verification

All 17 scripts were checked against their argparse definitions:

| # | Script | Status |
|---|--------|--------|
| 1 | `scripts/run_manifest_parallel.py` | MATCH |
| 2 | `scripts/run_manifest_baselines.py` | MATCH |
| 3 | `scripts/aggregate_results.py` | MATCH |
| 4 | `scripts/make_latex_table.py` | MATCH |
| 5 | `scripts/run_experiment.py` | MATCH |
| 6 | `scripts/synth_generate_s5.py` | MATCH |
| 7 | `scripts/sanity_check_v2_s5.py` | MATCH |
| 8 | `scripts/paper_quickstart.py` | MATCH |
| 9 | `scripts/run_all_experiments.py` | MATCH |
| 10 | `ext1/scripts/exp1_audit_story_leakage.py` | MATCH |
| 11 | `ext1/scripts/exp2_downstream_task_validation.py` | MATCH |
| 12 | `ext2/scripts/exp3_per_class_categorical.py` | MATCH |
| 13 | `ext2/scripts/exp4_shap_comparison.py` | MATCH |
| 14 | `ext2/scripts/exp4b_attention_rollout.py` | MATCH |
| 15 | `ext2/scripts/exp4c_d_stability.py` | MATCH |
| 16 | `ext2/scripts/exp5_significance_tests.py` | MATCH |
| 17 | `ext2/scripts/exp6_mimic_mortality_impute_predict.py` | MATCH |

**0 mismatches found.** All arguments in readme_CL.txt match their respective scripts.

### Task 3.2: One-Command Script Check

| Check | Status |
|-------|--------|
| `run_all_experiments.py` stages 1-7 match Task 2.1 table | PASS |
| `--dry-run` documented in README | PASS |
| `--skip-existing` documented for resume | PASS |
| Stage definitions match readme_CL.txt commands | PASS |

**Bug fixed:** `scripts/split_manifest_runner.py` line 81 referenced deleted `scripts/run_manifest.py`.
Changed to `scripts/run_manifest_parallel.py`.

**Bug fixed:** `docs/full_reproduction.md` line 78 used `--synth-root` but the script defines `--data-dir`.
Changed to `--data-dir`.

---

## Phase 4: Reproduction Environment

### Task 4.1: Installation & Environment

| Check | Status |
|-------|--------|
| Python 3.10+ requirement stated | PASS |
| conda env creation command | PASS |
| `pip install -r requirements_core.txt` | PASS |
| `pip install -r requirements_extra.txt` | PASS |
| GPU/PyTorch installation note | FIXED (was missing) |

**Fix applied:** Added GPU installation note to README.md Quick Install section, linking to
the official PyTorch installation page.

**Dependency completeness:** All third-party imports (numpy, pandas, scipy, scikit-learn,
torch, matplotlib, pyyaml, joblib, tqdm, networkx, tabulate, seaborn, shap, xgboost,
hyperimpute) are covered by `requirements_core.txt` or `requirements_extra.txt`.
No missing dependencies.

### Task 4.2: Dataset Documentation

| Check | Status |
|-------|--------|
| Dataset sources mentioned | PASS |
| Dataset table in README | FIXED (wrong counts) |
| Pre-generated missing data included in repo | PASS |
| `utility_missing_data_gen_v1/` documented | FIXED (added to structure) |

**Fix applied:** Corrected dataset table in README.md:

| Dataset | Old Samples | Actual Samples | Old Features | Actual Features |
|---------|-------------|----------------|--------------|-----------------|
| MIMIC | 5,000 | 2,052 | 8 | 8 |
| eICU | 5,000 | 1,430 | 9 | 20 |
| NHANES | 7,500 | 2,274 | 9 | 12 |
| Concrete | 1,030 | 1,030 | 9 | 9 |
| ComCri | 2,000 | 1,994 | 8 | 10 |
| AutoMPG | 392 | 392 | 8 | 8 |

Also fixed ComCri domain from "Materials Science" to "Social Science".

---

## Phase 5: Text Quality & Formatting

| Check | Status |
|-------|--------|
| Terminology consistency (SNI, CPFA, v0.3) | PASS |
| No v0.2 residuals in code | PASS |
| Code blocks have language tags (bash/python) | PASS |
| Tables render in GitHub Markdown | PASS |
| Documentation links valid | PASS |
| License badge links to correct license | PASS |
| readme_CL.txt section separators clear | PASS |
| Command line `\` continuation correct | PASS |
| Experiment summary table with runtimes | PASS |

---

## Phase 6: Security & Privacy

| Check | Status |
|-------|--------|
| No passwords/tokens/API keys | PASS |
| No hardcoded absolute paths in code | PASS |
| No `dengou` or `waseda` in tracked files | PASS |
| No SSH keys or certificates | PASS |
| No PII in data files (synthetic/anonymized) | PASS |
| `.gitignore` completeness | FIXED (was minimal) |
| `LICENSE` file exists | FIXED (was missing) |

**Fix applied:** Enhanced `.gitignore` to exclude:
- `results_*/`, `results/` (large experiment outputs)
- `*.egg-info/`, `.eggs/`, `*.egg`
- `.venv/`, `venv/`
- `.vscode/`, `.idea/`
- `.env`, `.env.local`

**Fix applied:** Created `LICENSE` file (MIT License) at repository root, matching the
MIT badge in README.md.

---

## Summary of All Changes

### Files Modified

| File | Change |
|------|--------|
| `README.md` | Fixed dataset table (samples/features), added `utility_missing_data_gen_v1/` to structure, added GPU note |
| `.gitignore` | Extended with result dirs, eggs, venvs, IDE, env files |
| `docs/full_reproduction.md` | Fixed `--synth-root` to `--data-dir` |
| `scripts/split_manifest_runner.py` | Fixed reference from deleted `run_manifest.py` to `run_manifest_parallel.py` |

### Files Created

| File | Description |
|------|-------------|
| `LICENSE` | MIT License file |
| `review_report.md` | This report |

### Items Requiring User Confirmation

1. **LICENSE copyright holder**: Currently set to "SNI Authors". Update to actual author name(s) before publishing.

2. **Citation block**: README.md has placeholder `author={...}, journal={...}`. Update with actual publication details.

3. **Dataset counts discrepancy**: The original README listed much larger sample counts (MIMIC: 5,000 vs actual 2,052; eICU: 5,000 vs 1,430; NHANES: 7,500 vs 2,274). This was corrected to match the actual CSV files in the repo. If the paper uses different (larger) datasets that are not included in the repo, the README should note this.
