import re
import pandas as pd
import shap
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from rules import RISK_RULES


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Terms Risk Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.big-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
}

.small-text {
    color: #94a3b8;
    margin-bottom: 18px;
    font-size: 17px;
}

.box-safe {
    background: linear-gradient(90deg, #14532d, #166534);
    color: white;
    padding: 16px;
    border-radius: 16px;
    border: 1px solid #22c55e;
    font-weight: 700;
    font-size: 20px;
}

.box-risk {
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    color: white;
    padding: 16px;
    border-radius: 16px;
    border: 1px solid #ef4444;
    font-weight: 700;
    font-size: 20px;
}

.reason-box {
    background: rgba(59,130,246,0.18);
    border: 1px solid rgba(59,130,246,0.35);
    color: #dbeafe;
    padding: 15px;
    border-radius: 14px;
    margin-top: 10px;
    margin-bottom: 12px;
}

.summary-safe {
    background: linear-gradient(90deg, #14532d, #166534);
    color: white;
    padding: 16px;
    border-radius: 16px;
    font-weight: 700;
    margin-top: 12px;
}

.summary-risk {
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    color: white;
    padding: 16px;
    border-radius: 16px;
    font-weight: 700;
    margin-top: 12px;
}

.clause-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 18px;
    font-size: 18px;
    line-height: 1.7;
}

.stButton > button {
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_user_text(text):
    text = str(text).lower().strip()

    replacements = {
        " ur ": " your ",
        " u ": " you ",
        "accses": "access",
        "acess": "access",
        "locaion": "location",
        "locatoin": "location",
        "permision": "permission",
        "galary": "gallery",
        "camra": "camera",
        "contcts": "contacts"
    }

    text = f" {text} "
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def clean_display_sentence(sentence):
    sentence = str(sentence).strip()
    sentence = re.sub(r'^\s*(\d+\s+)+', '', sentence)
    return sentence


def check_rules(sentence):
    sentence = normalize_user_text(sentence)
    matches = []

    for category, rule in RISK_RULES.items():
        matched_keywords = []

        for keyword in rule["keywords"]:
            if keyword.lower() in sentence:
                matched_keywords.append(keyword)

        if matched_keywords:
            matches.append({
                "risk_type": category,
                "risk_level": rule["risk_level"],
                "weight": rule["weight"],
                "matched_keywords": matched_keywords,
                "explanation": rule["explanation"]
            })

    if not matches:
        return {
            "has_rule_risk": False,
            "rule_risk_type": None,
            "rule_risk_level": None,
            "rule_reason": None,
            "matched_keywords": []
        }

    best_match = sorted(
        matches,
        key=lambda x: (x["weight"], len(x["matched_keywords"])),
        reverse=True
    )[0]

    return {
        "has_rule_risk": True,
        "rule_risk_type": best_match["risk_type"],
        "rule_risk_level": best_match["risk_level"],
        "rule_reason": best_match["explanation"],
        "matched_keywords": best_match["matched_keywords"]
    }


@st.cache_resource
def load_and_train():
    df = pd.read_csv("dataset_labeled.csv")

    df = df.dropna(subset=["text", "binary_label", "risk_type"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000
    )

    X = vectorizer.fit_transform(df["text"])

    # =========================
    # Binary model
    # =========================
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

    # =========================
    # Multi-class model
    # =========================
    df_multi = df[df["binary_label"].astype(str).str.lower() == "risk"].copy()

    X_multi = vectorizer.transform(df_multi["text"])

    risk_encoder = LabelEncoder()
    y_multi = risk_encoder.fit_transform(df_multi["risk_type"])

    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi,
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

    feature_names = vectorizer.get_feature_names_out()
    binary_explainer = shap.LinearExplainer(log_model, X_train_bin[:100])

    return (
        df,
        vectorizer,
        log_model,
        rf_model,
        binary_encoder,
        risk_encoder,
        binary_explainer,
        feature_names
    )


def shap_explainer_function(x, risk_prob, binary_encoder, binary_explainer, feature_names, top_k=5):
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
        reason_text = "ML flagged this clause because of terms like: " + ", ".join(
            [name for name, _ in reasons]
        )
    else:
        reason_text = "ML flagged this clause as risky, but no strong terms were identified."

    chart_df = pd.DataFrame(reasons, columns=["term", "impact"])
    return risk_level, reason_text, chart_df


def analyze_one_sentence_hybrid(sentence, vectorizer, log_model, rf_model, binary_encoder, risk_encoder, binary_explainer, feature_names, top_k=5):
    clean_sentence = normalize_user_text(sentence)

    # Rule-based check
    rule_result = check_rules(clean_sentence)

    # ML check
    x = vectorizer.transform([clean_sentence])

    pred_idx = int(log_model.predict(x)[0])
    pred_label = binary_encoder.inverse_transform([pred_idx])[0]

    classes = list(binary_encoder.classes_)
    risk_idx = classes.index("risk") if "risk" in classes else 1
    risk_prob = float(log_model.predict_proba(x)[0][risk_idx])

    ml_risk = str(pred_label).lower() == "risk"
    final_risk = rule_result["has_rule_risk"] or ml_risk

    result = {
        "sentence": clean_sentence,
        "predicted_label": pred_label,
        "risk_probability": risk_prob,
        "safe": not final_risk
    }

    if not final_risk:
        return result

    ml_risk_level, ml_reason, chart_df = shap_explainer_function(
        x, risk_prob, binary_encoder, binary_explainer, feature_names, top_k=top_k
    )

    risk_type_idx = int(rf_model.predict(x)[0])
    ml_risk_type = risk_encoder.inverse_transform([risk_type_idx])[0]

    if rule_result["has_rule_risk"]:
        final_risk_type = rule_result["rule_risk_type"]
    else:
        final_risk_type = ml_risk_type

    level_rank = {"Low": 1, "Medium": 2, "High": 3}
    rule_level = rule_result["rule_risk_level"] if rule_result["has_rule_risk"] else "Low"
    final_risk_level = rule_level if level_rank[rule_level] >= level_rank[ml_risk_level] else ml_risk_level

    if rule_result["has_rule_risk"]:
        rule_keywords = ", ".join(rule_result["matched_keywords"])
        final_reason = (
            f"Rule-based detection found risky indicators such as: {rule_keywords}. "
            f"{rule_result['rule_reason']}"
        )
    else:
        final_reason = ml_reason

    result["risk_level"] = final_risk_level
    result["risk_type"] = final_risk_type
    result["reason_text"] = final_reason
    result["chart_df"] = chart_df
    result["rule_triggered"] = rule_result["has_rule_risk"]
    result["rule_keywords"] = rule_result["matched_keywords"]

    return result


def check_risk(input_datapoint, vectorizer, log_model, rf_model, binary_encoder, risk_encoder, binary_explainer, feature_names, top_k=5):
    sentence = str(input_datapoint.iloc[0]["text"]).strip()
    return analyze_one_sentence_hybrid(
        sentence,
        vectorizer,
        log_model,
        rf_model,
        binary_encoder,
        risk_encoder,
        binary_explainer,
        feature_names,
        top_k=top_k
    )


def analyze_user_input(user_text, vectorizer, log_model, rf_model, binary_encoder, risk_encoder, binary_explainer, feature_names, top_k=5):
    sentences = split_into_sentences(user_text)
    results = []

    for sentence in sentences:
        results.append(
            analyze_one_sentence_hybrid(
                sentence,
                vectorizer,
                log_model,
                rf_model,
                binary_encoder,
                risk_encoder,
                binary_explainer,
                feature_names,
                top_k=top_k
            )
        )

    return results


def render_result(result):
    clean_sentence = clean_display_sentence(result["sentence"])

    st.markdown(
        f'<div class="clause-box">{clean_sentence}</div>',
        unsafe_allow_html=True
    )

    if result["safe"]:
        st.markdown('<div class="box-safe">✅ It\'s safe to go</div>', unsafe_allow_html=True)
        st.metric("Risk Probability", f"{result['risk_probability']:.2%}")
        return

    st.markdown('<div class="box-risk">⚠️ Not safe</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Probability", f"{result['risk_probability']:.2%}")
    c2.metric("Risk Level", result["risk_level"])
    c3.metric("Risk Type", result["risk_type"])

    st.markdown(f'<div class="reason-box"><b>Reason:</b> {result["reason_text"]}</div>', unsafe_allow_html=True)

    if result.get("rule_triggered"):
        st.caption("Rule-based layer was triggered before ML decision.")

    if "chart_df" in result and not result["chart_df"].empty:
        fig = px.bar(
            result["chart_df"].sort_values("impact"),
            x="impact",
            y="term",
            orientation="h",
            title="Top Risk-Contributing Terms",
            text="impact",
            color_discrete_sequence=["#ef4444"]
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            height=420,
            xaxis_title="Impact Score",
            yaxis_title="Term",
            title_x=0.02,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(result["chart_df"], use_container_width=True)


# =========================
# Load everything
# =========================
(
    df,
    vectorizer,
    log_model,
    rf_model,
    binary_encoder,
    risk_encoder,
    binary_explainer,
    feature_names
) = load_and_train()


# =========================
# UI
# =========================
st.markdown('<div class="big-title">🔍 Terms & Privacy Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="small-text">Hybrid AI system for clause-level risk detection, ranking, and explanation.</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📂 Analyze Dataset Row", "✍️ Analyze Your Own Text"])

with tab1:
    st.subheader("Dataset Clause Analysis")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        filter_mode = st.radio(
            "Dataset filter",
            ["All rows", "Risk rows only"],
            horizontal=True
        )

    if filter_mode == "Risk rows only":
        filtered_df = df[df["binary_label"].astype(str).str.lower() == "risk"].reset_index(drop=True)
    else:
        filtered_df = df.reset_index(drop=True)

    with col_b:
        row_index = st.number_input(
            "Choose row index",
            min_value=0,
            max_value=len(filtered_df) - 1,
            value=0,
            step=1
        )

    input_datapoint = filtered_df.iloc[[row_index]][["text"]]

    result = check_risk(
        input_datapoint,
        vectorizer,
        log_model,
        rf_model,
        binary_encoder,
        risk_encoder,
        binary_explainer,
        feature_names,
        top_k=5
    )

    render_result(result)

with tab2:
    st.subheader("Analyze Your Own Text")

    user_text = st.text_area(
        "Paste Terms / Privacy text هنا",
        height=220,
        placeholder="Example: We may share your data with third-party partners. The app may access your location and contacts."
    )

    top_k = 5

    if st.button("Analyze Text", use_container_width=False):
        if not user_text.strip():
            st.warning("من فضلك اكتب text الأول")
        else:
            results = analyze_user_input(
                user_text,
                vectorizer,
                log_model,
                rf_model,
                binary_encoder,
                risk_encoder,
                binary_explainer,
                feature_names,
                top_k=top_k
            )

            total_sentences = len(results)
            risky_sentences = sum(1 for r in results if not r["safe"])

            c1, c2 = st.columns(2)
            c1.metric("Total Clauses", total_sentences)
            c2.metric("Risky Clauses", risky_sentences)

            if risky_sentences == 0:
                st.markdown('<div class="summary-safe">Overall Result: The input looks safe.</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="summary-risk">Overall Result: {risky_sentences} risky clause(s) detected.</div>',
                    unsafe_allow_html=True
                )

            for i, result in enumerate(results, start=1):
                with st.expander(f"Clause {i}", expanded=(i == 1)):
                    render_result(result)