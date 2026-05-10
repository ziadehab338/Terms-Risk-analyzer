import re
def preprocess_text(text):
    """
    Preprocess the input text by removing extra whitespace and normalizing it.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text):
    """
    Split text into sentences using punctuation marks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]