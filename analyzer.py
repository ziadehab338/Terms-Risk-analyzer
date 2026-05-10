
from rules import RISK_RULES
from utils import preprocess_text, split_into_sentences
from PIL import Image
import easyocr
import numpy as np

def analyze_sentence(sentence):
    results = []
    sentence = preprocess_text(sentence)
    sentence_lower = sentence.lower()

    for category, rule in RISK_RULES.items():
        for keyword in rule["keywords"]:
            if keyword in sentence_lower:
                results.append({
                    "sentence": sentence,
                    "risk_type": category,
                    "risk_level": rule["risk_level"],
                    "explanation": rule["explanation"],
                    "matched_keyword": keyword
                })
    if not results:
        print("No risks detected.")

    return results


def analyze_text(sentences):
    all_results = []

    for sentence in sentences:
        sentence_results = analyze_sentence(sentence)
        if sentence_results:
            all_results.extend(sentence_results)
    if not all_results:
        print("No risks detected.")
    return all_results


def analyze_input(text):
    text = preprocess_text(text)
    sentences = split_into_sentences(text)

    if len(sentences) == 1:
        return analyze_sentence(sentences[0])
    else:
        return analyze_text(sentences)
    _reader = None

_reader = None

def get_ocr_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    reader = get_ocr_reader()
    ocr_results = reader.readtext(image_np, detail=0, paragraph=True)

    extracted_text = " ".join(ocr_results).strip()
    return extracted_text