# 小工具包（两个核心文件 + README）, additional comments for README.md

on Dec 31, 2025

## File structure
utility_missing_data_gen_v1/
  ├─ missing_data_generator.py     # 功能程序（核心：MCAR/MAR/MNAR + 校准 + 列类型）
  ├─ generate_missing_data.py      # 主控程序（CLI：参数化批量生成 + 命名 + 导出）
  └─ README.md

## 分为主控程序和功能程序
- 功能程序：missing_data_generator.py
   - 负责：列类型推断/覆盖、propensity（缺失倾向概率）构造、mask 采样、缺失率校准、输出 metadata
- 主控程序：generate_missing_data.py
   - 负责：解析命令行参数、批量组合（机制×缺失率）、保存 CSV（以及可选 mask/metadata）

## 主控程序参数
```bash
--input：输入 CSV 路径
--mechanisms：MCAR MAR MNAR
--dataset：数据集名（用于文件命名）
--rates：0.1 0.3 0.5
--output-dir：输出目录

--seed：随机种子（复现实验必备）
--exclude-cols：排除列（比如 ID/subject_id 不应被置缺失）
--categorical-cols / --continuous-cols：显式指定列类型（强烈推荐，避免“整数编码类别被当成连续”）
--mar-driver-cols：MAR 依赖的驱动列（默认自动取前2个连续列）
--tolerance：总体缺失率允许误差（默认 0.01）
--min-missing-per-col：强制每列至少有多少缺失（默认 1，避免某些列 0 缺失导致评估不稳）
--rate-style：缺失率在文件名中的编码风格（见下）
```

## 输出 CSV 命名规则
```bash
{dataset}_{mechanism}_{rateTag}.csv
```
其中 rateTag 默认是 10per / 30per / 50per，例如：
	•	ComCri_MCAR_30per.csv
	•	Concrete_MNAR_50per.csv

如果更希望文件名里直接出现 0.3（便于脚本解析），可用：
--rate-style float

生成：
	•	ComCri_MCAR_0.3.csv

如果你想“既像 0.3 又避免点号”，用：
--rate-style p
生成：
	•	ComCri_MCAR_0p3.csv

(4) 其他我认为必须优化的部分 ✅（已经做了）
	1.	缺失率校准更审稿友好
原脚本为了达成目标缺失率，会“随机补缺/随机恢复”，会破坏 MAR/MNAR 结构。
新版本改为：用 propensity（倾向概率）排序 做校准：

	•	缺失不够：优先把“更该缺”的位置置缺
	•	缺失过多：优先恢复“最不该缺”的位置
这样更能保持机制一致性（审稿更安全）。

	2.	列类型可控（很关键）
很多你们的数据里，类别列可能是整数编码（例如 SpO2=97/98/99/100）。
如果只靠 pandas dtype，会误判为连续列，进而 MNAR 机制/后续实验都会偏。
所以主控程序支持显式传入列名列表。
	3.	可选输出 mask 和 metadata（强烈推荐）
你们后续要“重跑全套实验确保可信”，mask+metadata 能极大提高可追溯性。
可选输出：

	•	*_mask.npy：精确缺失位置（True=missing），完全可复现
	•	*_meta.json：记录 seed、target/actual 缺失率、列类型、每列缺失率、MAR driver cols、MNAR 参数等

⸻

典型用法示例：

1) 一次生成三种机制 × 三种缺失率（推荐）+显式指定类别列（强烈推荐）
```bash
python generate_missing_data.py \
  --input /path/to/Concrete.csv \
  --output-dir /path/to/out \
  --dataset Concrete \
  --mechanisms MCAR MAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "XXX,XXX" \
  --save-mask --save-metadata
```

2) 排除 ID 列
```bash
python generate_missing_data.py \
  --input /path/to/MIMIC.csv \
  --output-dir /path/to/out \
  --dataset MIMIC \
  --exclude-cols "subject_id,hadm_id,stay_id" \
  --mechanisms MAR \
  --rates 0.3 \
  --seed 2025 \
  --save-mask --save-metadata
  ```

3) 显式指定类别列（强烈推荐：SpO2 这种）
```bash
python generate_missing_data.py \
  --input /path/to/MIMIC.csv \
  --output-dir /path/to/out \
  --dataset MIMIC \
  --categorical-cols "SpO2,ALARM" \
  --mechanisms MNAR \
  --rates 0.3 \
  --seed 2025 \
  --save-mask --save-metadata
  ```

## 后续实验衔接的实用建议
为了后续“一键跑所有结果 CSV + 一键出出版级图表”，建议在实验主控里这样组织：
- 预处理阶段：固定 seed 生成每个 {dataset, mechanism, rate} 的 missing CSV + mask.npy + meta.json
- 实验阶段：所有 imputation 方法（SNI / SNI-M / baselines）读取同一份 mask.npy（或者至少同一份缺失 CSV）
- 这样各方法比较才是“同一缺失模式”上的严格对比，也更容易写进论文（审稿信任度更高）

⸻

（Option）未来可考虑再做一个小扩展：
支持 --seeds 1 2 3 5 8 一次性生成多份 missing pattern（并在文件名里带 seed），用于“5 seeds 的缺失模式也一起随机”的更严谨设置——也可以在这个工具上直接加上（改动很小）。


---
实施（2025-12-31）

### 先加上ID列

python utility_missing_data_gen_v1/01_add_ID.py \
  --input data/raw.csv \
  --output data/with_id.csv


### AutoMPG
python utility_missing_data_gen_v1/generate_missing_data.py \
  --input data/AutoMPG_complete.csv \
  --output-dir data/AutoMPG/ \
  --dataset AutoMPG \
  --mechanisms MAR MCAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "model_year,origin" \
  --exclude-cols ID \
  --save-mask --save-metadata

### ComCri
python utility_missing_data_gen_v1/generate_missing_data.py \
  --input data/ComCri_complete.csv \
  --output-dir data/ComCri/ \
  --dataset ComCri \
  --mechanisms  MNAR MAR MCAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "IncomeLevel,UrbanType,EducationLevel,CrimeLevel,RegionCode" \
  --exclude-cols ID \
  --save-mask --save-metadata 

### Concrete
python utility_missing_data_gen_v1/generate_missing_data.py \
  --input data/Concrete_complete.csv \
  --output-dir data/Concrete/ \
  --dataset Concrete \
  --mechanisms MCAR MAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "" \
  --exclude-cols ID \
  --save-mask --save-metadata 

### eICU
python utility_missing_data_gen_v1/generate_missing_data.py \
  --input data/eICU_complete.csv \
  --output-dir data/eICU/ \
  --dataset eICU \
  --mechanisms MNAR MCAR MAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "mechanical_ventilation_std,vasopressor_use_std,age_band,gender_std,composite_risk_score" \
  --exclude-cols ID \
  --save-mask --save-metadata \
  --overwrite


### MIMIC
python utility_missing_data_gen_v1/generate_missing_data.py \
  --input data/MIMIC_complete.csv \
  --output-dir data/MIMIC/ \
  --dataset MIMIC \
  --mechanisms MCAR MAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "SpO2,ALARM" \
  --exclude-cols ID \
  --save-mask --save-metadata 

### NHANES
python utility_missing_data_gen_v1/generate_missing_data.py \
  --input data/NHANES_complete.csv \
  --output-dir data/NHANES/ \
  --dataset NHANES \
  --mechanisms MCAR MAR MNAR \
  --rates 0.1 0.3 0.5 \
  --seed 2025 \
  --categorical-cols "gender_std,age_band" \
  --exclude-cols ID \
  --save-mask --save-metadata


#### NHANES_metabolic_score_NoMis 所有列名
waist_circumference,systolic_bp,diastolic_bp,triglycerides,hdl_cholesterol,fasting_glucose,age,bmi,hba1c,gender_std,age_band,metabolic_score

#### NHANES notes:
【门控特征】
gender_std: 性别 (0=男性, 1=女性)
age_band: 年龄分段 (0=18-29, 1=30-39, 2=40-49, 3=50-59, 4=60-69, 5=70+)
bp_med_std: 降压药使用 (0=未使用, 1=使用)
lipid_med_std: 降脂药使用 (0=未使用, 1=使用)
glucose_med_std: 降糖药使用 (0=未使用, 1=使用)
smoking_std: 吸烟状态 (0=不吸烟, 1=吸烟)
fasting_state_std: 空腹状态 (0=非空腹<8h, 1=空腹≥8h)

# NHANES_metabolic_syndrome_NoMis 所有列名
waist_circumference,systolic_bp,diastolic_bp,triglycerides,hdl_cholesterol,fasting_glucose,age,bmi,hba1c,gender_std,age_band,metabolic_syndrome

目标变量 (y)
-----------
metabolic_score: 代谢综合征评分 (0-5, 异常组分数量) - 回归任务
metabolic_syndrome: 代谢综合征诊断 (0/1, ≥3项异常) - 分类任务


## 检查哪些missing.csv的ID有问题：

python - <<'PY'
import glob
import pandas as pd

bad = []
for fp in glob.glob("data/*/*_*.csv"):
    df = pd.read_csv(fp)
    if "ID" not in df.columns:
        continue
    na = df["ID"].isna().sum()
    dup = df["ID"].duplicated().sum()
    if na > 0 or dup > 0:
        bad.append((fp, na, dup))
if not bad:
    print("[OK] All missing CSVs have clean unique ID.")
else:
    print("[BAD] ID problems found:")
    for fp, na, dup in bad:
        print(f" - {fp}: NA={na}, DUP={dup}")
PY