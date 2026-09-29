# 3. Proste czyszczenie tekstu
import re
from sklearn.model_selection import train_test_split

lemma_map = {
    "drukarki": "drukarka",
    "drukarce": "drukarka",
    "drukarkę": "drukarka",
    "funkcjonalności": "funkcjonalność",
    "ustawien": "ustawienia",
}


def apply_lemma(text):
    for key, val in lemma_map.items():
        text = text.replace(key, val)
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text


def clean_input(input):
    return [apply_lemma(clean_text(t)) for t in input]


def split_data(input, labels, test_size=0.3):
    return train_test_split(
        input, labels, test_size=test_size, random_state=42, stratify=labels
    )
