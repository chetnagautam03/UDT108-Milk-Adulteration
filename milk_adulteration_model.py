# UDT108 - Project and Case Study
# Project Title: Milk Adulteration Detection Using Machine Learning (NIR Spectral Data)
# Programme: B.Sc. (H) Chemistry (Research) | K.R. Mangalam University

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. LOAD AND PREPROCESS DATA
print("Loading NIR Spectral Data...")
csv_files = glob.glob('*.csv')
data_frames = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Assign labels based on filenames
        if 'pure' in file.lower() and 'powder' not in file.lower() and 'flour' not in file.lower() and 'sugar' not in file.lower() and 'coffee' not in file.lower():
            label = 'Pure Milk'
        elif 'sugar' in file.lower():
            label = 'Adulterated (Sugar)'
        elif 'flour' in file.lower():
            label = 'Adulterated (Flour)'
        elif 'coffee' in file.lower():
            label = 'Adulterated (Coffee)'
        else:
            label = 'Other Adulterated/Powder'
            
        df['Label'] = label
        data_frames.append(df)
    except Exception as e:
        pass

# Combine all valid files
full_df = pd.concat(data_frames, ignore_index=True)

# Separate features (wavelengths) and target (Label)
X = full_df.select_dtypes(include=[np.number]) # NIR Spectral readings
y = full_df['Label']

print(f"Total Samples Loaded: {X.shape[0]}")
print(f"Spectral Features (Wavelengths): {X.shape[1]}")
print("Class Distribution:\n", y.value_counts())

# 2. TRAIN-TEST SPLIT & SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. MODEL TRAINING & EVALUATION
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1
    })

results_df = pd.DataFrame(results)
print("\n--- MODEL PERFORMANCE ---")
print(results_df.to_string(index=False))

# 4. VISUALISATION: Model Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Classification Accuracy of Machine Learning Models')
plt.ylim(0, 1.1)
for i, v in enumerate(results_df['Accuracy']):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.savefig('model_comparison.png', dpi=300)
print("\nSaved 'model_comparison.png'")

# 5. VISUALISATION: Confusion Matrix for Random Forest
rf_model = models["Random Forest"]
y_pred_rf = rf_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Saved 'confusion_matrix.png'")
