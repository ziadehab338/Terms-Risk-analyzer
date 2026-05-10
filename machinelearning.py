import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# =========================
# 1) Load data
# =========================
df = pd.read_csv("dataset_labeled.csv")

# تنظيف بسيط
df = df.dropna(subset=["text", "binary_label", "risk_type"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# =========================
# 2) TF-IDF for text
# =========================
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=30000
)
X = vectorizer.fit_transform(df["text"])
binary_encoder = LabelEncoder()
y_binary = binary_encoder.fit_transform(df["binary_label"])

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X,
    y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_model.fit(X_train_bin, y_train_bin)

y_pred_bin = log_model.predict(X_test_bin)

print("\n" + "="*50)
print("Binary Classification (Logistic Regression)")
print("="*50)
print("Accuracy:", accuracy_score(y_test_bin, y_pred_bin))
print(classification_report(
    y_test_bin,
    y_pred_bin,
    target_names=binary_encoder.classes_
))

risk_encoder = LabelEncoder()
y_multi = risk_encoder.fit_transform(df["risk_type"])

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X,
    y_multi,
    test_size=0.2,
    random_state=42,
    stratify=y_multi
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train_multi, y_train_multi)

y_pred_multi = rf_model.predict(X_test_multi)

print("\n" + "="*50)
print("Multi-Class Classification (Random Forest)")
print("="*50)
print("Accuracy:", accuracy_score(y_test_multi, y_pred_multi))
print(classification_report(
    y_test_multi,
    y_pred_multi,
    target_names=risk_encoder.classes_
))
print("Start SHAP for Logistic Regression...")
input_datapoint = df.iloc[[911]][["text"]]
feature_names = vectorizer.get_feature_names_out()
binary_explainer = shap.LinearExplainer(log_model, X_train_bin[:100])

def shap_explainer_function(x, risk_prob, top_k=5):
    classes = list(binary_encoder.classes_)
    risk_idx = classes.index("risk") if "risk" in classes else 1

    if risk_prob >= 0.80:
        risk_level = "High"
    elif risk_prob >= 0.50:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    exp = binary_explainer(x)
    values = exp.values

    if values.ndim == 3:
        shap_row = values[0, :, risk_idx]
    else:
        shap_row = values[0]

    present_idx = x.nonzero()[1]

    reasons = []
    for idx in present_idx:
        contribution = float(shap_row[idx])
        if contribution > 0:
            reasons.append((feature_names[idx], contribution))

    reasons = sorted(reasons, key=lambda z: z[1], reverse=True)[:top_k]

    if reasons:
        reason_text = ", ".join([f"{name} ({value:+.3f})" for name, value in reasons])
    else:
        reason_text = "No strong reasons found"

    if reasons:
     reason_text = f"The model marked this sentence as risky because of terms like: {', '.join([name for name, value in reasons])}"
    else:
       reason_text = "The model marked this sentence as risky, but no strong terms were found."

    print("Reason:", reason_text)  


def check_risk(input_datapoint, top_k=5):
    sentence = str(input_datapoint.iloc[0]["text"]).strip()
    x = vectorizer.transform([sentence])

    pred_idx = int(log_model.predict(x)[0])
    pred_label = binary_encoder.inverse_transform([pred_idx])[0]

    classes = list(binary_encoder.classes_)
    risk_idx = classes.index("risk") if "risk" in classes else 1
    risk_prob = float(log_model.predict_proba(x)[0][risk_idx])

    print("Sentence from dataset:")
    print(sentence)
    print("Predicted Label:", pred_label)
    print("Risk Probability:", f"{risk_prob:.2%}")
    print("-" * 50)

    if str(pred_label).lower() != "risk":
        print("It's safe to go")
        return

    shap_explainer_function(x, risk_prob, top_k)
check_risk(input_datapoint)    