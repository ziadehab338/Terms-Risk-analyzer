# 🔍 Terms Risk Detector

An intelligent system designed to analyze Terms of Service and Privacy Policies, uncover hidden risks, and present them in a clear and understandable way.

---

## 🚨 Problem

Most users accept Terms of Service without reading them. These documents are often long, complex, and intentionally difficult to understand, which can expose users to hidden risks such as data sharing, automatic payments, or loss of rights.

---

## 💡 Solution

This project introduces a **Risk Detection System** that audits legal text and transforms it into actionable insights.

The system:
- Breaks down legal text into individual clauses
- Detects potentially dangerous statements
- Classifies the type of risk
- Explains why each clause is risky in simple language

---

## 🧠 System Architecture

The system is built using a hybrid approach:

### 1️⃣ Logic-Based Risk Checker
- Uses a predefined **risk ontology**
- Detects “Danger Clauses” using keywords and patterns
- Provides transparent and explainable results

### 2️⃣ Machine Learning Model
- Multi-class classification model
- Predicts the type of risk in each clause
- Learns semantic patterns beyond keywords

### 3️⃣ Comparative Analysis
- Compares rule-based vs AI-based detection
- Evaluates accuracy, interpretability, and coverage

---

## ⚙️ Features

- 🔎 Clause-level analysis
- ⚠️ Risk classification (High / Medium / Low)
- 🧾 Risk type detection:
  - Data Sharing
  - Privacy & Tracking
  - Payment & Auto-Renewal
  - Permissions Access
  - Legal Liability
- 💬 Human-readable explanations
- 📊 Risk scoring system

---

## 📌 Example

**Input:**
We may share your data with third-party partners.

**Output:**
- Risk Level: High  
- Risk Type: Data Sharing  
- Explanation: This clause allows sharing user data with external parties.

---

## 🛠️ Technologies

- Python
- Natural Language Processing (NLP)
- scikit-learn
- Rule-based systems
- (Planned) Transformers / LLMs

---

## 📂 Project Structure

terms-risk-detector/
│
├── app.py              # Main entry point
├── rules.py            # Risk ontology and rules
├── analyzer.py         # Clause analysis logic
├── utils.py            # Text processing utilities
├── requirements.txt
└── README.md

---

## 🚀 Future Improvements

- PDF document analysis
- Image-to-text (OCR) support
- Advanced ML models (BERT / Transformers)
- Explainability tools (SHAP, LIME)
- Web-based user interface

---

## 🎯 Impact

This project aims to bridge the gap between complex legal language and everyday users by making hidden risks visible, understandable, and actionable.

---