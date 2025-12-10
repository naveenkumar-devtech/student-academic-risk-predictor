# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import joblib

# ------------- Load dataset -------------
# Make sure student-mat.csv is in the same folder as this script
df = pd.read_csv("student-mat.csv", sep=";")  # UCI version uses ; as separator

print("Initial shape:", df.shape)

# ------------- Target: define 'at risk' -------------
# G3 is final grade (0-20). We'll mark at-risk if G3 < 10.
df = df.copy()
df["target"] = (df["G3"] < 10).astype(int)  # 1 = at-risk, 0 = low-risk

print("Class distribution:")
print(df["target"].value_counts())

# ------------- Feature engineering -------------

# Create average past grade feature
df["avg_past_grade"] = df[["G1", "G2"]].mean(axis=1)

# Approximate attendance score (higher absences -> worse attendance)
max_absences = df["absences"].max() if df["absences"].max() > 0 else 1
df["attendance_rate"] = 1 - (df["absences"] / max_absences)

# Numerical features
num_features = [
    "age",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "avg_past_grade",
    "attendance_rate",
]

# Categorical features (subset of important ones)
cat_features = [
    "sex",        # F, M
    "address",    # U, R
    "famsize",    # LE3, GT3
    "Pstatus",    # T, A
    "schoolsup",  # yes, no
    "famsup",     # yes, no
    "paid",       # yes, no
    "activities", # yes, no
    "higher",     # yes, no
    "internet",   # yes, no
]

# Drop rows with missing values in selected columns
df = df.dropna(subset=num_features + cat_features + ["target"])

X = df[num_features + cat_features]
y = df["target"]

print("Shape after selecting features:", X.shape)

# ------------- Preprocessing pipeline -------------

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# ------------- Train / test split -------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------- Model 1: Logistic Regression -------------

log_reg = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)

print("\n=== Logistic Regression ===")
print(classification_report(y_test, log_preds, digits=3))
log_acc = accuracy_score(y_test, log_preds)
log_prec = precision_score(y_test, log_preds)
log_rec = recall_score(y_test, log_preds)
print(f"Accuracy: {log_acc:.3f}, Precision: {log_prec:.3f}, Recall: {log_rec:.3f}")

# ------------- Model 2: Random Forest -------------

rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            max_depth=None,
        )),
    ]
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_preds, digits=3))
rf_acc = accuracy_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds)
rf_rec = recall_score(y_test, rf_preds)
print(f"Accuracy: {rf_acc:.3f}, Precision: {rf_prec:.3f}, Recall: {rf_rec:.3f}")

# ------------- Choose final model -------------

# Prefer better recall; if similar, prefer better accuracy
if rf_rec > log_rec:
    final_model = rf
    chosen = "Random Forest"
else:
    final_model = log_reg
    chosen = "Logistic Regression"

print(f"\nChosen final model: {chosen}")

# ------------- Save final model -------------

joblib.dump(final_model, "model.joblib")
print("Saved trained model to model.joblib")
