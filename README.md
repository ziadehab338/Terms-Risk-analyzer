# 🔍 Explainable AI for Terms & Privacy Risk Detection

An advanced intelligent system designed to analyze **Terms of Service** and **Privacy Policies**, uncover hidden risks, and transform complex legal language into clear, interpretable, and actionable insights.

This project addresses a critical real-world problem by combining rule-based reasoning with machine learning models to deliver both transparency and predictive power.

---

## 🚨 The Problem

Modern digital platforms rely heavily on long, complex legal agreements that users rarely read.

These documents are intentionally dense and difficult to interpret, which often leads to users unknowingly accepting risky conditions such as:

- Unauthorized data sharing with third parties
- Hidden subscription and automatic payment mechanisms
- Extensive tracking and privacy violations
- Legal clauses limiting user rights and protections

As a result, there is a significant gap between user awareness and actual risk exposure.

---

## 💡 The Solution

This project introduces an **Explainable Risk Detection System** that analyzes legal text at the clause level and converts it into meaningful insights.

Instead of simply detecting risk, the system focuses on:

- Understanding why a clause is risky
- Explaining it in plain, human-readable language
- Providing structured and interpretable outputs

The system transforms unstructured legal text into:

- Risk classifications
- Risk scores
- Clear explanations
- Actionable recommendations

---

## 🧠 System Architecture

The system follows a **hybrid AI architecture**, combining deterministic logic with data-driven learning.

### 1️⃣ Rule-Based Risk Engine

- Built on a predefined risk ontology
- Detects dangerous clauses using keyword matching and pattern recognition
- Provides fully transparent and explainable decisions
- Serves as a baseline for comparison

### 2️⃣ Machine Learning Models

- **Binary Classification Model**: Predicts whether a clause is risky or not (using interpretable models like Logistic Regression).
- **Multi-Class Classification Model**: Classifies the type of risk, including:
  - Data Sharing
  - Privacy & Tracking
  - Payments & Subscriptions
  - Permissions Access
  - Legal Liability

### 3️⃣ Comparative Intelligence Layer & Semantic AI

- Compares rule-based vs ML outputs
- Integrates **Semantic AI** to understand the meaning and context of clauses, capturing deeper insights beyond keyword matching.
- Highlights differences in:
  - Accuracy
  - Coverage
  - Interpretability
  - **Meaningful Interpretation**: Using semantic analysis to interpret complex legal clauses.
  
This hybrid approach allows a deeper understanding of trade-offs between explainability, predictive performance, and the meaning behind legal texts.Captures semantic patterns beyond keywords.

## ⚙️ Core Features

- 🔎 Clause-level text analysis
- ⚠️ Risk level classification (High / Medium / Low)
- 🧾 Multi-category risk detection
- 💬 Explainable outputs (not black-box predictions)
- 📊 Risk scoring mechanism
- 🔁 Hybrid detection (Rule-based + ML)
- 🧠 Semantic understanding of legal text
- 🖼️ **Image Upload and Analysis**: Allows users to upload images for OCR-based text extraction and analysis.
- 📝 **Text Upload**: Users can directly input text for risk assessment and analysis.
- 🖥️ **Interactive User Interface**: A user-friendly interface for easy interaction, including text and image input for seamless risk detection.

---

## 🛠️ Technologies Used

- **Python**
- **Natural Language Processing (NLP)**
- **scikit-learn**
- **Rule-based systems**
- **Streamlit (Interactive Interface)**
- (Planned) **Transformer-based models (BERT, LLMs)**

---

## 📂 Project Structure
terms-risk-detector/
├── app.py                  # Streamlit application interface
├── rules.py                # Risk ontology and detection rules
├── analyzer.py             # Core analysis engine
├── semantic_ai.py          # Semantic AI for clause detection
├── utils.py                # Text preprocessing & segmenting
├── models/                 # Saved ML models
├── requirements.txt        # Project dependencies
└── README.md               # Documentation

---
## 🌐 Live Demo

You can access the live demo here: [Streamlit Demo](https://terms-risk-analyzer-dlfqvbrwk87qly7wgf9tkd.streamlit.app)
