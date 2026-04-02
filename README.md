# RTscore — Chromatographic Plausibility Scoring for LC

A tool for evaluating whether candidate molecular structures are **chromatographically plausible** based on the relationship between **molecular descriptors** and **observed retention time (RT)** or **retention index (RI)**.

## Web App

The app is implemented as a **Streamlit interface** and is designed to support **metabolomics annotation workflows**.

You can use the application directly online:

👉 **[Launch RTscore](https://rt-score.streamlit.app/)**

No installation is required.

---

# Table of Contents

1. Concept
2. Why chromatographic plausibility matters
3. Installation
4. Input files
5. Running the analysis
6. Interpreting each tab
7. Understanding the suspicion score
8. Critical interpretation of results
9. Best practices
10. Limitations
11. Integration with metabolomics pipelines
12. Example workflow

---

# 1. Concept

When identifying compounds in LC-MS metabolomics, the usual workflow relies on:

* accurate mass
* isotope pattern
* MS/MS fragmentation
* database matches

However, these signals **do not guarantee chromatographic consistency**.

Two molecules with the same mass may have **very different retention behavior**.

This tool evaluates whether a candidate structure is **compatible with the chromatographic system** by comparing:

```
Observed RT or RI
vs
Predicted RT or RI
```

The prediction is derived from **molecular descriptors** such as:

* logP
* TPSA
* H-bond donors
* H-bond acceptors
* rotatable bonds
* aromatic rings
* molecular weight

These descriptors are calculated automatically using **RDKit**.

---

# 2. Why Chromatographic Plausibility Matters

Many metabolomics annotations fail because the structure:

* has correct mass
* matches a database
* but **elutes in an impossible chromatographic region**

Examples:

| Compound           | Expected behavior | Observed RT |
| ------------------ | ----------------- | ----------- |
| Highly polar sugar | early             | late        |
| Hydrophobic lipid  | late              | early       |

Such mismatches can indicate:

* wrong candidate
* incorrect adduct
* spectral contamination
* co-elution
* database bias

Chromatographic plausibility acts as a **third orthogonal filter** after MS1 and MS/MS.

---

# 3. Installation

Clone the repository:

```bash
git clone https://github.com/yourrepo/RTscore
cd RTscore
```

Create environment:

```bash
conda create -n rtscore python=3.11
conda activate rtscore
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies:

```
streamlit
pandas
numpy
plotly
rdkit
pillow
```

Run the application:

```bash
streamlit run app_rtscore.py
```

---

# 4. Input Files

Two CSV files are required.

## Reference Dataset

This dataset defines the **chromatographic behavior of known compounds**.

Required columns:

| column   | description               |
| -------- | ------------------------- |
| name     | compound name             |
| smiles   | molecular structure       |
| rt or ri | chromatographic reference |

Optional metadata:

```
class
mode
adduct
```

Example:

```csv
name,smiles,rt,class
Caffeine,Cn1cnc2n(C)c(=O)n(C)c(=O)c12,1.92,alkaloid
Quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.25,flavonoid
```

### Critical comment

The reference dataset **defines the model domain**.

A poor reference dataset leads to a poor model.

Good reference datasets should:

* represent the chemical diversity of the study
* contain ≥20 compounds
* cover the RT range of interest

---

## Candidate Dataset

Contains candidate structures for each MS feature.

Required columns:

| column         | description         |
| -------------- | ------------------- |
| feature_id     | MS feature          |
| candidate_name | candidate structure |
| smiles         | molecular structure |

Optional:

```
observed_rt
observed_ri
candidate_class
rank_source
```

Example:

```csv
feature_id,candidate_name,smiles,observed_rt
F001,Quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.22
```

---

## Calibrants Dataset (optional)

Only required when using **Retention Index (RI)** mode.

Columns:

```
rt,index
```

Example:

```csv
rt,index
1.2,100
2.1,200
3.4,300
4.5,400
```

These compounds are used to **interpolate RI values**.

---

# 5. Running the Analysis

Steps inside the interface:

### Step 1 — Upload files

Upload:

* reference dataset
* candidate dataset

Optionally:

* calibrant dataset

---

### Step 2 — Choose prediction axis

Options:

```
Retention Time (RT)
Retention Index (RI)
```

### RT mode

Uses observed retention times directly.

### RI mode

If RI is not provided, the system computes it via:

```
RI interpolation from calibrants
```

Critical note:

Retention indices improve **cross-experiment reproducibility**.

---

### Step 3 — Select descriptors

Default descriptors include:

```
MolLogP
TPSA
HBD
HBA
RotatableBonds
AromaticRingCount
MolWt
```

Critical insight:

Descriptors reflect **chromatographic physicochemistry**.

Example influences:

| descriptor      | chromatographic effect     |
| --------------- | -------------------------- |
| logP            | hydrophobicity             |
| TPSA            | polarity                   |
| HBD             | hydrogen bonding           |
| Rotatable bonds | conformational flexibility |

Choosing irrelevant descriptors reduces model quality.

---

### Step 4 — Select model

Two models are available.

## Weighted descriptor score

Uses predefined weights based on chromatographic intuition.

Advantages:

* robust
* interpretable
* stable with small datasets

Recommended for small reference sets.

---

## Linear regression

Fits a regression model using the reference dataset.

Advantages:

* data-driven
* adaptive

Limitations:

* requires larger datasets
* risk of overfitting

---

### Step 5 — Set suspicion thresholds

Example:

```
Highly plausible < 1
Plausible < 2
Borderline < 3
Suspicious ≥ 3
```

These thresholds convert a **continuous score** into categories.

Critical insight:

Thresholds should reflect:

* chromatographic precision
* RT reproducibility
* column type

---

### Step 6 — Run analysis

Click:

```
Run analysis
```

The app will:

1. calculate molecular descriptors
2. fit the reference model
3. predict RT or RI
4. compute deviations
5. compute suspicion scores
6. rank candidates

---

# 6. Interpreting Each Tab

## Overview

Shows:

* model type
* axis used
* residual standard deviation
* dataset summaries

Critical question:

```
Is the reference dataset adequate?
```

If residual SD is very large:

```
model may be unreliable
```

---

## Reference Model

Two diagnostic plots.

### Observed vs Predicted

Ideal pattern:

```
points near diagonal
```

Warning signs:

```
systematic deviation
large scatter
```

---

### Residual plot

Shows model errors.

Interpretation:

```
large residuals → model mismatch
clustered residuals → descriptor bias
```

---

## Prediction View

Displays:

```
candidate predicted vs observed RT
```

And the **candidate table**.

Key columns:

| column          | meaning                   |
| --------------- | ------------------------- |
| abs_error       | chromatographic deviation |
| suspicion_score | normalized deviation      |
| applicability   | descriptor distance       |

---

## Candidate Plausibility

Most important tab.

Three visualizations:

### Reference score distribution

Shows where the candidate lies relative to known compounds.

Good candidates:

```
near center of distribution
```

---

### Feature plausibility map

Axes:

```
x = predicted RT
y = suspicion score
```

Bubble size:

```
chemical distance from reference space
```

Best candidates:

```
low suspicion score
small distance
```

---

### Candidate ranking

Simple comparison between structures.

Lower bars:

```
better chromatographic consistency
```

---

## Structures

Displays chemical structures.

Useful for identifying patterns such as:

```
extra hydroxyl groups
ring substitutions
sugar moieties
```

These structural changes often explain RT shifts.

---

## Export

Exports:

```
reference results
candidate results
```

Useful for:

* reports
* manuscript figures
* integration with pipelines

---

# 7. Suspicion Score

The suspicion score is defined as:

```
abs_error / reference_residual_sd
```

Meaning:

```
how many standard deviations the candidate deviates from expected RT
```

Example:

| score | interpretation  |
| ----- | --------------- |
| 0.5   | very consistent |
| 1.5   | acceptable      |
| 2.5   | questionable    |
| 4     | very suspicious |

---

# 8. Critical Interpretation

Chromatographic plausibility **does not prove identity**.

It only answers:

```
Is this structure chromatographically consistent?
```

A candidate should ideally satisfy:

```
accurate mass ✓
MS/MS match ✓
chromatographic plausibility ✓
```

---

# 9. Best Practices

Use:

* ≥20 reference compounds
* compounds covering RT range
* consistent LC conditions

Avoid:

* mixing columns
* mixing gradients
* mixing mobile phases

---

# 10. Limitations

Important limitations include:

### Column dependence

Different LC columns change RT relationships.

---

### Gradient dependence

Different gradients change elution behavior.

---

### Chemical space coverage

If the candidate lies outside the reference space:

```
model predictions become unreliable
```

This is why the **applicability domain metric** is calculated.

---

# 11. Integration with Metabolomics Workflows

RTscore fits naturally after:

```
MS1 annotation
MS/MS annotation
```

Pipeline example:

```
feature detection
→ formula prediction
→ spectral library match
→ RTscore plausibility filter
→ final annotation ranking
```

---

# 12. Example Workflow

Example metabolomics identification pipeline:

```
LC-HRMS data
↓
feature detection (MZmine)
↓
candidate structures (GNPS / SIRIUS / database)
↓
RTscore filtering
↓
manual inspection
↓
annotation level assignment
```

---

# Final Recommendation

RTscore should be used as a **decision support tool**, not an automatic identifier.

The strongest annotations combine:

```
spectral evidence
chromatographic plausibility
chemical reasoning
```
