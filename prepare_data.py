import pandas as pd
import os
import re
import pandas as pd
from rules import RISK_RULES

DATA_FOLDER = r"C:\Users\DELL\Downloads\OPP-115\pretty_print"

data = []

def split_sentences(text):
    sentences = re.split(r'[.!?]\s+', str(text))
    return sentences

def is_valid_sentence(text):
    text = str(text).strip()
    words = text.split()

    if len(text) < 15:
        return False

    if "@" in text:
        return False

    if len(words) == 1:
        return False

    risk_keywords = [
        "share", "collect", "data", "personal", "liable",
        "terminate", "access", "permission", "track",
        "location", "camera", "microphone", "contacts",
        "photos", "gallery", "payment", "subscription"
    ]

    if len(words) < 6:
        if any(word in text.lower() for word in risk_keywords):
            return True
        return False

    return True


for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATA_FOLDER, filename)

        try:
            df_file = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        except:
            continue
        if "text" in df_file.columns:
            texts = df_file["text"].dropna().astype(str).tolist()

        else:
            texts = []
            for _, row in df_file.iterrows():
                row_text = " ".join(row.dropna().astype(str))
                texts.append(row_text)

        for text in texts:
            sentences = split_sentences(text)

            for sentence in sentences:
                if is_valid_sentence(sentence):
                    data.append({
                        "file": filename,
                        "text": sentence.strip()
                    })
df = pd.DataFrame(data)
df.to_csv("dataset_raw.csv", index=False)
print("Dataset created successfully!")
print(f"Total sentences: {len(df)}")
df = pd.read_csv("dataset_raw.csv")
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

def auto_label_sentence(text):
    sentence = str(text).strip()
    sentence_lower = sentence.lower()

    matches = []

    for category, rule in RISK_RULES.items():
        matched_keywords = []

        for keyword in rule["keywords"]:
            if keyword.lower() in sentence_lower:
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
        return pd.Series({
            "binary_label": "no_risk",
            "risk_type": "No Risk",
            "risk_level": "Low",
            "risk_score": 0,
            "matched_keywords": "",
            "explanation": "No risky clause detected by rule-based labeling."
        })

    best_match = sorted(
        matches,
        key=lambda x: (x["weight"], len(x["matched_keywords"])),
        reverse=True
    )[0]
    all_risk_types = list(set([m["risk_type"] for m in matches]))
    all_keywords = []
    for m in matches:
        all_keywords.extend(m["matched_keywords"])

    return pd.Series({
        "binary_label": "risk",
        "risk_type": best_match["risk_type"],
        "risk_level": best_match["risk_level"],
        "risk_score": best_match["weight"],
        "matched_keywords": ", ".join(sorted(set(all_keywords))),
        "explanation": best_match["explanation"],
        "all_risk_types": ", ".join(all_risk_types)
    })

labels_df = df["text"].apply(auto_label_sentence)
df_labeled = pd.concat([df, labels_df], axis=1)
df_labeled = df_labeled.drop_duplicates(subset=["text"])
df_labeled.to_csv("dataset_labeled.csv", index=False)