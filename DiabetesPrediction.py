import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay)

# 1. Load Dataset
print(" PROJECT  — Diabetes Prediction")
print("\nLoading Pima Indians Diabetes dataset...")

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(URL, header=None, names=COLUMNS)

X = df.drop("Outcome", axis=1)
y = df["Outcome"].values

print(f" Samples : {X.shape[0]}")
print(f" Features : {X.shape[1]}")
print(f"\nFeature names:\n  {list(X.columns)}")
print(f"\nTarget distribution:")
print(f" No Diabetes (0): {(y==0).sum()}")
print(f" Diabetes (1): {(y==1).sum()}")

# 2. Handle Impossible Zero Values
X = X.copy()
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print(f"\nReplacing impossible zeros with NaN:")
for col in cols_with_zeros:
    zeros = (X[col] == 0).sum()
    print(f"  {col}: {zeros} zeros → NaN")
    X[col] = X[col].replace(0, np.nan)

# 3. Feature Engineering
print("\nFeature Engineering")

X["bmi_category"] = pd.cut(
    X["BMI"],
    bins=[0, 18.5, 25, 30, 100],
    labels=[0, 1, 2, 3]).astype(float)
print(" + bmi_category (0=Underweight 1=Normal 2=Overweight 3=Obese)")

# Glucose risk level
X["glucose_risk"] = pd.cut(
    X["Glucose"],
    bins=[0, 100, 125, 500],
    labels=[0, 1, 2]).astype(float)
print(" + glucose_risk (0=Normal 1=Prediabetes 2=Diabetes range)")

# Age group
X["age_group"] = pd.cut(
    X["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]).astype(float)
print(" + age_group (0=Young 1=Middle 2=Senior 3=Elderly)")

# Insulin-glucose ratio
X["insulin_glucose"] = X["Glucose"] / (X["Insulin"] + 1e-6)
print(" + insulin_glucose")

# BMI × Glucose combined risk
X["bmi_glucose"] = X["BMI"] * X["Glucose"] / 1000
print(" + bmi_glucose")

# Blood pressure adjusted for age
X["bp_age"] = X["BloodPressure"] / (X["Age"] + 1e-6)
print(" + bp_age")

print(f"\nFinal feature count: {X.shape[1]}")

# 4. Show impact of feature engineering
original_cols = COLUMNS[:-1] 
X_original = X[original_cols].copy()

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score_before = cross_val_score(
    rf, scaler.fit_transform(imputer.fit_transform(X_original)), y,cv=cv, scoring="roc_auc").mean()

score_after = cross_val_score(rf, scaler.fit_transform(imputer.fit_transform(X)), y,cv=cv, scoring="roc_auc").mean()

print(f"\nRandom Forest AUC — original features  : {score_before:.3f}")
print(f"Random Forest AUC — engineered features: {score_after:.3f}")
print(f"Improvement: +{score_after - score_before:.3f}")

# 5. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Define Models
def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)])

models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    "KNN":                 make_pipeline(KNeighborsClassifier(n_neighbors=9)),
    "SVM":                 make_pipeline(SVC(kernel="rbf", C=10, probability=True, random_state=42)),
    "Random Forest":       make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
    "XGBoost":             make_pipeline(XGBClassifier(n_estimators=100, eval_metric="logloss",
                                                        random_state=42, verbosity=0))}

# 7. Train and Evaluate All Models
results = {}

print("\n" + "=" * 55)
print(" MODEL COMPARISON")
print("=" * 55)

for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "test_acc": pipeline.score(X_test, y_test),
        "auc":      auc,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
    }

    print(f"\n{name}")
    print(f" CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f" Test Acc : {pipeline.score(X_test, y_test):.3f}")
    print(f" AUC-ROC : {auc:.3f}")
    print(classification_report(y_test, y_pred,target_names=["No Diabetes", "Diabetes"],zero_division=0))

best_name = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest model by AUC: {best_name} ({results[best_name]['auc']:.3f})")

# 8. Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Project — Diabetes Prediction", fontsize=14, fontweight="bold")

# Plot 1: AUC comparison
names = list(results.keys())
auc_vals = [results[n]["auc"] for n in names]
colors = ["#27ae60" if n == best_name else "#82e0aa" for n in names]
axes[0, 0].barh(names, auc_vals, color=colors, edgecolor="white", height=0.6)
axes[0, 0].set_xlabel("AUC-ROC Score")
axes[0, 0].set_title("AUC-ROC Comparison")
axes[0, 0].set_xlim(0.6, 1.0)
for i, v in enumerate(auc_vals):
    axes[0, 0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

# Plot 2: Confusion matrix
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(
    ax=axes[0, 1], colorbar=False, cmap="Greens")
axes[0, 1].set_title(f"Confusion Matrix — {best_name}")

# Plot 3: ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[1, 0].plot(fpr, tpr, label=f"{name} ({res['auc']:.2f})", linewidth=1.5)
axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].set_title("ROC Curves — All Models")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Feature importances (red = engineered, blue = original)
rf_pipeline = models["Random Forest"]
rf_model = rf_pipeline.named_steps["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
colors_feat = ["#c0392b" if f not in original_cols else "#85c1e9" for f in importances.index]
axes[1, 1].barh(importances.index, importances.values, color=colors_feat)
axes[1, 1].set_xlabel("Importance Score")
axes[1, 1].set_title("Feature Importances\n(red = engineered, blue = original)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("project_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as project_results.png")

# 9. Feature Engineering Impact Chart
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    ["Original\n(8 features)", "Engineered\n(14 features)"],
    [score_before, score_after],
    color=["#85c1e9", "#27ae60"], width=0.4, edgecolor="white")
ax.set_ylabel("AUC-ROC (Random Forest, 5-fold CV)")
ax.set_title("Impact of Feature Engineering on Model Performance")
ax.set_ylim(0.7, 0.9)
for bar, val in zip(bars, [score_before, score_after]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
            f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("project_feature_impact.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature impact chart saved as project_feature_impact.png")