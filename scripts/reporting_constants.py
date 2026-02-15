"""Centralized constants for all reporting / visualization scripts."""

# ─── Algorithm ordering & display names ───
ALGO_ORDER_MAIN = [
    "MeanMode", "KNN", "MICE", "MissForest", "GAIN", "MIWAE", "SNI"
]
ALGO_ORDER_SUPP = ALGO_ORDER_MAIN[:-1] + ["HyperImpute", "TabCSDI", "SNI"]

ALGO_RENAME_MAP = {
    "MeanMode": "Mean/Mode",
}

# ─── Dataset ordering ───
DATASET_ORDER = [
    "MIMIC", "eICU", "NHANES", "ComCri", "AutoMPG", "Concrete"
]

# ─── Color palette (algorithm -> color) ───
ALGO_COLORS = {
    "MeanMode": "#999999", "KNN": "#377eb8", "MICE": "#4daf4a",
    "MissForest": "#984ea3", "GAIN": "#ff7f00", "MIWAE": "#a65628",
    "HyperImpute": "#f781bf", "TabCSDI": "#e41a1c", "SNI": "#1b9e77",
}

# ─── Metric display helpers ───
METRIC_DIRECTION = {
    "cont_NRMSE": "min", "cont_MAE": "min", "cont_R2": "max",
    "cat_Accuracy": "max", "cat_Macro-F1": "max", "cat_Kappa": "max",
    "runtime_sec": "min",
}
