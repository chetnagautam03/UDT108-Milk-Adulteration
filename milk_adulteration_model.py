# ==============================================================================
# UDT108 PROJECT: MILK ADULTERATION DETECTION USING NEAR-INFRARED (NIR) SPECTROSCOPY
# Student Name : Chetna Ahuja
# Roll Number  : 2303210015
# Course       : UDT108 - Project and Case Study (Semester VII)
# Institution  : School of Basic and Applied Sciences (SBAS), K.R. Mangalam University
# Dataset DOI  : 10.17632/3gxjgxkg76.1 (Mendeley Data NIR Spectral Repository)
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix

def apply_snv(X_matrix):
    """Standard Normal Variate (SNV) scatter correction for turbid milk emulsions."""
    mean_row = np.mean(X_matrix, axis=1, keepdims=True)
    std_row = np.std(X_matrix, axis=1, keepdims=True)
    return (X_matrix - mean_row) / (std_row + 1e-8)

def preprocess_nir_spectra(df_wavelengths):
    """Executes SNV scatter removal followed by Savitzky-Golay 1st-derivative filtering."""
    X_snv = apply_snv(df_wavelengths.values)
    X_sg1 = savgol_filter(X_snv, window_length=11, polyorder=2, deriv=1, axis=1)
    return X_sg1

def main():
    print("[INFO] Loading Mendeley NIR Milk Adulteration Dataset...")
    df = pd.read_csv('Mendeley_NIR_Milk_Adulteration_900_1700nm.csv')
    
    wavelength_cols = [c for c in df.columns if c not in ['Sample_ID', 'Adulterant_Class', 'Conc_wv']]
    
    print(f"[INFO] Spectral Range: 900.0 nm to 1700.0 nm ({len(wavelength_cols)} Continuous Channels)")
    X_processed = preprocess_nir_spectra(df[wavelength_cols])
    y_encoded = LabelEncoder().fit_transform(df['Adulterant_Class'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print(f"[INFO] Training Set Shape: {X_train_sc.shape} | Test Set Shape: {X_test_sc.shape}")
    
    classifiers = {
        "Random Forest (n=300, Gini)": RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_split=3, criterion='gini', random_state=42, n_jobs=-1
        ),
        "Support Vector Machine (RBF)": SVC(
            kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42
        ),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='minkowski', p=2
        ),
        "Logistic Regression (L2)": LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42
        )
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    print("\n" + "="*95)
    print("TRAINING AND BENCHMARKING SUPERVISED CLASSIFIERS (5-FOLD STRATIFIED CV)")
    print("="*95)
    
    for name, clf in classifiers.items():
        cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=cv_strategy, scoring='accuracy')
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)
        y_prob = clf.predict_proba(X_test_sc)
        
        acc = accuracy_score(y_test, y_pred) * 100
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro') * 100
        
        results.append({
            "Model": name,
            "Test Accuracy (%)": round(acc, 2),
            "5-Fold CV Mean (%)": f"{round(cv_scores.mean()*100, 2)} ± {round(cv_scores.std()*100, 2)}",
            "Macro Precision (%)": round(p*100, 2),
            "Macro Recall (%)": round(r*100, 2),
            "Macro F1-Score (%)": round(f1*100, 2),
            "ROC-AUC (%)": round(auc, 2)
        })
        print(f"-> Completed Model: {name:<30} | Test Accuracy: {acc:5.2f}% | ROC-AUC: {auc:5.2f}%")
        
    df_results = pd.DataFrame(results)
    print("="*95)
    print(df_results.to_string(index=False))
    print("="*95)
    
    rf_best = classifiers["Random Forest (n=300, Gini)"]
    importances = rf_best.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    
    print("\n--- TOP 5 DIAGNOSTIC NIR WAVELENGTHS (RANDOM FOREST GINI IMPORTANCE) ---")
    for rank, idx in enumerate(top_idx, 1):
        print(f"Rank {rank}: Wavelength = {wavelength_cols[idx]} nm | Gini Importance = {importances[idx]:.4f}")
        
    print("\n--- RANDOM FOREST CLASSIFICATION REPORT ---")
    y_pred_rf = rf_best.predict(X_test_sc)
    print(classification_report(y_test, y_pred_rf, digits=3))

if __name__ == "__main__":
    main()
