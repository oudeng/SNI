# Reviewer Risk Checklist

This checklist addresses potential reviewer concerns about reproducibility and rigor.

## Checklist

- [x] **Version ambiguity eliminated**: Only SNI v0.3 remains in the repository. `grep -rn "SNI_v0_2" --include="*.py"` returns empty.

- [x] **Reproduction barrier lowered**: Three-line QuickStart available. `--smoke-test` mode verifies the full pipeline in minutes on CPU.

- [x] **Result overload controlled**: Profile system (`configs/paper_profiles.yaml`) separates main paper (7 algorithms) from supplement (9 algorithms). Scripts accept `--profile` to generate targeted outputs.

- [x] **Baseline fairness transparent**: Budget parameters (timeout, epochs, diffusion_steps) are explicitly recorded in `metrics_summary.json`. Self-test script (`baseline_selftest.py`) validates implementations.

- [x] **Dependencies stabilized**: `requirements_core.txt` (main paper) and `requirements_extra.txt` (supplements) are split. `environment_snapshot.txt` is auto-generated at runtime.

## Verification Commands

```bash
# 1. No v0.2 residuals
grep -rn "SNI_v0_2" --include="*.py" | grep -v ".git/"
# Expected: empty

# 2. Smoke test passes
python scripts/paper_quickstart.py --smoke-test --outdir /tmp/sni_smoke
# Expected: "Smoke test PASSED"

# 3. Profile system works
python -c "from scripts.profile_utils import load_profile; print(load_profile('configs/paper_profiles.yaml:main'))"
# Expected: dict with include_algos, datasets, etc.

# 4. Baseline self-test
python scripts/baseline_selftest.py
# Expected: sanity check results printed

# 5. Environment snapshot
python -c "from scripts.env_snapshot import write_environment_snapshot; from pathlib import Path; write_environment_snapshot(Path('/tmp/sni_env'))"
ls /tmp/sni_env/environment_snapshot.txt
# Expected: file exists
```
