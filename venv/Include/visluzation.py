import os
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
from analyzer import extract_text_from_image
from semantic_ai import build_semantic_index, semantic_check


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Terms Risk Analyzer",
    page_icon="🔍",
    layout="wide"
)


# =========================
# File Path Helper
# =========================
def find_project_file(filename):
    """
    Search for a file upward from the current file directory.
    This fixes problems when the app is accidentally run from venv/Include.
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))

    for _ in range(10):
        candidate = os.path.join(current_dir, filename)

        if os.path.exists(candidate):
            return candidate

        parent_dir = os.path.dirname(current_dir)

        if parent_dir == current_dir:
            break

        current_dir = parent_dir

    raise FileNotFoundError(
        f"Could not find {filename}. Make sure it exists inside your project folder."
    )


# =========================
# CSS
# =========================
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

.hero {
    min-height: 430px;
    padding: 58px;
    border-radius: 34px;
    background:
        radial-gradient(circle at top left, rgba(244,63,94,0.28), transparent 28%),
        radial-gradient(circle at bottom right, rgba(37,99,235,0.35), transparent 32%),
        linear-gradient(135deg, #111827 0%, #020617 55%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 24px 80px rgba(0,0,0,0.42);
    margin-bottom: 34px;
}

.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.16);
    color: #bfdbfe;
    border: 1px solid rgba(59,130,246,0.35);
    border-radius: 999px;
    padding: 8px 14px;
    font-weight: 700;
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 56px;
    font-weight: 950;
    margin-bottom: 18px;
    line-height: 1.1;
}

.hero p {
    font-size: 20px;
    color: #dbeafe;
    max-width: 900px;
    line-height: 1.8;
}

.hero-small {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 18px;
}

.feature-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    padding: 24px;
    border-radius: 22px;
    min-height: 165px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.20);
}

.feature-card h4 {
    margin-bottom: 12px;
    font-size: 21px;
    font-weight: 800;
}

.feature-card p {
    color: #b6c5d8;
    font-size: 16px;
    line-height: 1.65;
}

.analysis-panel {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 28px;
    border-radius: 26px;
    margin-top: 22px;
}

.mode-title {
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 8px;
}

.mode-subtitle {
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 24px;
}

.stButton > button {
    border-radius: 16px;
    padding: 0.75rem 1.4rem;
    font-weight: 800;
    font-size: 17px;
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


# =========================
# Model Loading / Training
# =========================
@st.cache_resource
def load_and_train(data_path):
    df = pd.read_csv(data_path)

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

    # Binary model
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

    # Multi-class model
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


@st.cache_resource
def load_semantic_layer(_df):
    build_semantic_index(_df)
    return True


# =========================
# SHAP Explanation
# =========================
def shap_explainer_function(
    x,
    risk_prob,
    binary_encoder,
    binary_explainer,
    feature_names,
    top_k=5
):
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


# =========================
# Hybrid Analyzer
# =========================
def analyze_one_sentence_hybrid(
    sentence,
    vectorizer,
    log_model,
    rf_model,
    binary_encoder,
    risk_encoder,
    binary_explainer,
    feature_names,
    top_k=5
):
    clean_sentence = normalize_user_text(sentence)

    # Rule-based layer
    rule_result = check_rules(clean_sentence)

    # ML binary layer
    x = vectorizer.transform([clean_sentence])

    pred_idx = int(log_model.predict(x)[0])
    pred_label = binary_encoder.inverse_transform([pred_idx])[0]

    classes = list(binary_encoder.classes_)
    risk_idx = classes.index("risk") if "risk" in classes else 1
    risk_prob = float(log_model.predict_proba(x)[0][risk_idx])

    ml_risk = str(pred_label).lower() == "risk"

    # Semantic AI layer
    semantic_result = semantic_check(clean_sentence, top_k=5, threshold=0.55)
    semantic_risk = semantic_result["semantic_risk"]

    # Final decision
    final_risk = rule_result["has_rule_risk"] or ml_risk or semantic_risk

    result = {
        "sentence": clean_sentence,
        "predicted_label": pred_label,
        "risk_probability": risk_prob,
        "safe": not final_risk,
        "rule_triggered": rule_result["has_rule_risk"],
        "rule_keywords": rule_result["matched_keywords"],
        "semantic_triggered": semantic_risk,
        "semantic_score": semantic_result["semantic_score"],
        "semantic_matches": semantic_result["semantic_matches"]
    }

    if not final_risk:
        return result

    # ML explanation
    ml_risk_level, ml_reason, chart_df = shap_explainer_function(
        x,
        risk_prob,
        binary_encoder,
        binary_explainer,
        feature_names,
        top_k=top_k
    )

    # ML multi-class risk type
    risk_type_idx = int(rf_model.predict(x)[0])
    ml_risk_type = risk_encoder.inverse_transform([risk_type_idx])[0]

    # Final risk type priority
    if rule_result["has_rule_risk"]:
        final_risk_type = rule_result["rule_risk_type"]
    elif semantic_risk:
        final_risk_type = semantic_result["semantic_risk_type"]
    elif ml_risk:
        final_risk_type = ml_risk_type
    else:
        final_risk_type = ml_risk_type

    # Final risk level
    level_rank = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    rule_level = rule_result["rule_risk_level"] if rule_result["has_rule_risk"] else "Low"

    if semantic_risk and semantic_result["semantic_score"] >= 0.70:
        semantic_level = "High"
    elif semantic_risk:
        semantic_level = "Medium"
    else:
        semantic_level = "Low"

    candidate_levels = [rule_level, ml_risk_level, semantic_level]

    final_risk_level = max(
        candidate_levels,
        key=lambda level: level_rank.get(level, 1)
    )

    # Final explanation priority
    if rule_result["has_rule_risk"]:
        rule_keywords = ", ".join(rule_result["matched_keywords"])
        final_reason = (
            f"Rule-based detection found risky indicators such as: {rule_keywords}. "
            f"{rule_result['rule_reason']}"
        )
    elif semantic_risk:
        final_reason = semantic_result["semantic_reason"]
    elif ml_risk:
        final_reason = ml_reason
    else:
        final_reason = "No risky indicators were found."

    result["risk_level"] = final_risk_level
    result["risk_type"] = final_risk_type
    result["reason_text"] = final_reason
    result["chart_df"] = chart_df

    return result


def check_risk(
    input_datapoint,
    vectorizer,
    log_model,
    rf_model,
    binary_encoder,
    risk_encoder,
    binary_explainer,
    feature_names,
    top_k=5
):
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


def analyze_user_input(
    user_text,
    vectorizer,
    log_model,
    rf_model,
    binary_encoder,
    risk_encoder,
    binary_explainer,
    feature_names,
    top_k=5
):
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


def analyze_image(
    uploaded_file,
    vectorizer,
    log_model,
    rf_model,
    binary_encoder,
    risk_encoder,
    binary_explainer,
    feature_names,
    top_k=5
):
    extracted_text = extract_text_from_image(uploaded_file)

    if not extracted_text.strip():
        return [], ""

    results = analyze_user_input(
        extracted_text,
        vectorizer,
        log_model,
        rf_model,
        binary_encoder,
        risk_encoder,
        binary_explainer,
        feature_names,
        top_k=top_k
    )

    return results, extracted_text


# =========================
# Render Result
# =========================
def render_result(result):
    clean_sentence = clean_display_sentence(result["sentence"])

    st.markdown(
        f'<div class="clause-box">{clean_sentence}</div>',
        unsafe_allow_html=True
    )

    if result["safe"]:
        st.markdown(
            '<div class="box-safe">✅ It\'s safe to go</div>',
            unsafe_allow_html=True
        )
        st.metric("Risk Probability", f"{result['risk_probability']:.2%}")
        return

    st.markdown(
        '<div class="box-risk">⚠️ Not safe</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Risk Probability", f"{result['risk_probability']:.2%}")
    c2.metric("Risk Level", result["risk_level"])
    c3.metric("Risk Type", result["risk_type"])

    st.markdown(
        f'<div class="reason-box"><b>Reason:</b> {result["reason_text"]}</div>',
        unsafe_allow_html=True
    )

    if result.get("rule_triggered"):
        st.caption("Rule-based layer was triggered before ML decision.")

    if result.get("semantic_triggered"):
        st.caption(
            f"AI semantic layer was triggered. Similarity score: {result['semantic_score']:.2f}"
        )

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

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

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
# Load Everything
# =========================
DATA_PATH = find_project_file("dataset_labeled.csv")

(
    df,
    vectorizer,
    log_model,
    rf_model,
    binary_encoder,
    risk_encoder,
    binary_explainer,
    feature_names
) = load_and_train(DATA_PATH)

load_semantic_layer(df)


# =========================
# Session State
# =========================
if "show_analyzer" not in st.session_state:
    st.session_state.show_analyzer = False

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "✍️ Analyze Text"


# =========================
# Landing Page
# =========================
if not st.session_state.show_analyzer:

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Hybrid AI • Rule-Based + Machine Learning + Semantic AI + OCR</div>
        <h1>🔍 Terms & Privacy Risk Analyzer</h1>
        <p>
            Understand hidden risks inside Terms of Service and Privacy Policies.
            Paste text directly or upload a screenshot, and the system will analyze it,
            detect risky clauses, classify the risk type, rank the risk level, and explain
            why the clause may not be safe.
        </p>
        <div class="hero-small">
            Built for clause-level analysis, explainable results, semantic understanding,
            and privacy-risk awareness.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <h4>✍️ Paste or Type Text</h4>
            <p>
                Copy any Terms or Privacy clause and get an instant risk analysis with
                probability, level, type, and explanation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <h4>🖼️ Upload Screenshot</h4>
            <p>
                Upload a screenshot from an app or website. OCR extracts the text,
                then the analyzer checks it for hidden risks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <h4>🧠 Hybrid AI Detection</h4>
            <p>
                The system combines rules, machine learning, semantic similarity,
                and explanations to show why a clause is risky.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if st.button("🚀 Let’s Analyze Your Text", use_container_width=True):
        st.session_state.show_analyzer = True
        st.rerun()


# =========================
# Analyzer Page
# =========================
else:

    st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)

    top_left, top_right = st.columns([6, 3])

    with top_left:
        st.markdown(
            '<div class="mode-title">Start Your Analysis</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="mode-subtitle">Choose how you want to provide the content: write/paste text or upload a screenshot.</div>',
            unsafe_allow_html=True
        )

    with top_right:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.show_analyzer = False
            st.rerun()

    st.session_state.analysis_mode = st.radio(
        "Choose input method",
        ["✍️ Analyze Text", "🖼️ Upload Screenshot"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # =========================
    # Text Analysis
    # =========================
    if st.session_state.analysis_mode == "✍️ Analyze Text":

        st.subheader("✍️ Analyze Text")

        user_text = st.text_area(
            "Write, paste, or copy Terms / Privacy text here",
            height=240,
            placeholder=(
                "Example: We may share your data with third-party partners. "
                "The app may access your location and contacts."
            )
        )

        if st.button("Analyze Text", key="analyze_text_btn"):

            if not user_text.strip():
                st.warning("Please enter text first.")

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
                    top_k=5
                )

                total_sentences = len(results)
                risky_sentences = sum(1 for r in results if not r["safe"])

                m1, m2 = st.columns(2)

                m1.metric("Total Clauses", total_sentences)
                m2.metric("Risky Clauses", risky_sentences)

                if risky_sentences == 0:
                    st.markdown(
                        '<div class="summary-safe">Overall Result: The input looks safe.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="summary-risk">Overall Result: {risky_sentences} risky clause(s) detected.</div>',
                        unsafe_allow_html=True
                    )

                for i, result in enumerate(results, start=1):
                    with st.expander(f"Clause {i}", expanded=(i == 1)):
                        render_result(result)

    # =========================
    # Screenshot Analysis
    # =========================
    else:

        st.subheader("🖼️ Analyze Screenshot")

        uploaded_file = st.file_uploader(
            "Upload Terms / Privacy screenshot",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption="Uploaded Screenshot",
                use_container_width=True
            )

            if st.button("Analyze Screenshot", key="analyze_screenshot_btn"):

                image_results, extracted_text = analyze_image(
                    uploaded_file,
                    vectorizer,
                    log_model,
                    rf_model,
                    binary_encoder,
                    risk_encoder,
                    binary_explainer,
                    feature_names,
                    top_k=5
                )

                if not extracted_text.strip():
                    st.warning("No readable text was found in the image.")

                else:
                    st.subheader("Extracted Text")

                    st.text_area(
                        "OCR Result",
                        value=extracted_text,
                        height=180
                    )

                    total_sentences = len(image_results)
                    risky_sentences = sum(1 for r in image_results if not r["safe"])

                    m1, m2 = st.columns(2)

                    m1.metric("Total Clauses", total_sentences)
                    m2.metric("Risky Clauses", risky_sentences)

                    if risky_sentences == 0:
                        st.markdown(
                            '<div class="summary-safe">Overall Result: The screenshot looks safe.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="summary-risk">Overall Result: {risky_sentences} risky clause(s) detected.</div>',
                            unsafe_allow_html=True
                        )

                    for i, result in enumerate(image_results, start=1):
                        with st.expander(f"Screenshot Clause {i}", expanded=(i == 1)):
                            render_result(result)

    st.markdown('</div>', unsafe_allow_html=True)