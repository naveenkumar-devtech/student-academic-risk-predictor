# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained pipeline (preprocessor + classifier)
model = joblib.load("model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    explanation = None

    if request.method == "POST":
        # Read inputs from form (must match features in train.py)
        age = int(request.form.get("age", 17))
        Medu = int(request.form.get("Medu", 2))
        Fedu = int(request.form.get("Fedu", 2))
        studytime = int(request.form.get("studytime", 2))
        failures = int(request.form.get("failures", 0))
        G1 = float(request.form.get("G1", 10))
        G2 = float(request.form.get("G2", 10))
        absences = float(request.form.get("absences", 0))

        sex = request.form.get("sex", "F")
        address = request.form.get("address", "U")
        famsize = request.form.get("famsize", "GT3")
        Pstatus = request.form.get("Pstatus", "T")
        schoolsup = request.form.get("schoolsup", "no")
        famsup = request.form.get("famsup", "yes")
        paid = request.form.get("paid", "no")
        activities = request.form.get("activities", "no")
        higher = request.form.get("higher", "yes")
        internet = request.form.get("internet", "yes")

        # Derived features (same as train.py)
        avg_past_grade = (G1 + G2) / 2.0
        attendance_rate = 1 - (absences / (absences + 1))

        row = {
            "age": age,
            "Medu": Medu,
            "Fedu": Fedu,
            "studytime": studytime,
            "failures": failures,
            "avg_past_grade": avg_past_grade,
            "attendance_rate": attendance_rate,
            "sex": sex,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "higher": higher,
            "internet": internet,
        }

        X = pd.DataFrame([row])

        # Probability of class 1 (at-risk)
        proba = model.predict_proba(X)[0][1]
        label = "At-risk" if proba >= 0.5 else "Low-risk"

        result = {
            "label": label,
            "probability": float(proba),
        }

        # Try to extract simple explanation from RandomForest
        explanation = None
        try:
            clf = model.named_steps["classifier"]
            preproc = model.named_steps["preprocessor"]

            if hasattr(clf, "feature_importances_"):
                num_features = preproc.transformers_[0][2]
                cat_pipeline = preproc.transformers_[1][1]
                onehot = cat_pipeline.named_steps["onehot"]
                cat_cols = preproc.transformers_[1][2]
                cat_names = onehot.get_feature_names_out(cat_cols)
                feature_names = list(num_features) + list(cat_names)
                importances = clf.feature_importances_
                idx = np.argsort(importances)[::-1][:5]
                top = [(feature_names[i], float(importances[i])) for i in idx]
                explanation = top
        except Exception:
            explanation = None

    return render_template("predict.html", result=result, explanation=explanation)


@app.route("/help")
def help_page():
    return render_template("help.html")

if __name__ == "__main__":
    app.run()
