# Milk Adulteration Detection Using Machine Learning (NIR Spectroscopy 900–1700 nm)

**Course:** UDT108 – Project and Case Study (Semester VII)  
**Student:** Chetna Ahuja (Roll No: 2303210015)  
**Programme:** B.Sc. (Honours) Chemistry (Research) with Minor in Data Science  
**Institution:** School of Basic and Applied Sciences (SBAS), K.R. Mangalam University, Gurugram  
**Dataset Reference:** Mendeley Data NIR Spectral Repository (`DOI: 10.17632/3gxjgxkg76.1`)

## 1. Project Overview
This repository contains the end-to-end chemometric and supervised machine learning pipeline for rapid, non-destructive detection of synthetic and economic adulterants in liquid bovine milk (**Urea, Starch, Added Water, Hydrogen Peroxide, and Detergent/Bicarbonate**) across 228 continuous Near-Infrared (NIR) wavelength channels (**900.0 nm – 1700.0 nm**).

## 2. Chemometric Preprocessing Pipeline
To eliminate multiplicative Mie light scattering from milk fat globules and resolve overlapping O–H, N–H, and C–H overtone bands prior to classification:
1. **Standard Normal Variate (SNV):** Row-wise mean centering and unit-variance scaling of each spectrum.
2. **Savitzky-Golay 1st-Derivative Filtering:** Moving polynomial convolution (`window_length=11`, `polyorder=2`, `deriv=1`).
3. **Leakage-Free Partitioning:** Stratified 80:20 train-test split (`n_train = 960`, `n_test = 240`) with `StandardScaler` fitted strictly on the training matrix.

## 3. Model Benchmark Results (5-Fold Stratified CV & Test Set)

| Supervised ML Classifier | Test Accuracy | 5-Fold CV Mean | Macro Precision | Macro Recall | Macro F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (n=300, Gini)** | **98.50%** | **98.21% (±0.42%)** | **98.60%** | **98.30%** | **98.40%** | **99.60%** |
| Support Vector Machine (RBF) | 94.20% | 93.84% (±0.68%) | 94.40% | 93.90% | 94.00% | 97.80% |
| K-Nearest Neighbors (k=5) | 89.80% | 89.12% (±1.05%) | 90.10% | 89.20% | 89.50% | 94.10% |
| Logistic Regression (L2) | 85.40% | 84.70% (±1.28%) | 85.60% | 84.50% | 84.90% | 91.20% |

## 4. Top Diagnostic NIR Wavelengths (Random Forest Gini Importance)
* **1490.2 nm (Gini = 0.2450):** N–H 1st Overtone Symmetric/Asymmetric Stretch (Urea `[CO(NH2)2]` & Amides)
* **1455.1 nm (Gini = 0.2102):** O–H 1st Overtone Stretch (Water Dilution & Starch Hydrocolloids)
* **1204.8 nm (Gini = 0.1651):** C–H 2nd Overtone Stretch (Milk Triglycerides / Fat Globules)
* **975.0 nm (Gini = 0.1304):** O–H 2nd Overtone Stretch (Aqueous Moisture Baseline)
* **1430.5 nm (Gini = 0.1048):** C–O–H Combination Vibrational Band (Starch / Sucrose)

## 5. Repository Structure
* `UDT108_NIR_Milk_Adulteration_Chetna_2303210015.ipynb` – Google Colaboratory notebook with executed cells and logs.
* `milk_adulteration_model.py` – Standalone Python training, cross-validation, and subgroup auditing script.
* `fig1.png` to `fig5.png` – High-resolution (300 DPI) spectral, ROC, correlation, subgroup audit, and confusion matrix plots.
* `colab_screen_1.png` to `colab_screen_4.png` – Executed Google Colaboratory code & console output screenshots.
* `requirements.txt` – Pinned Python library dependencies.
