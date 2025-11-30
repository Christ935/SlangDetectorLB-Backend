import re
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from fuzzywuzzy import fuzz

# ---------------- NLP SETUP ----------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

DATASET_PATH = "lebanese_drug_slangv3.csv"
MODEL_PATH = "enhanced_multiclass_model.joblib"

english_stop = set(stopwords.words("english"))
leb_stopwords = {
    "ya", "wallah", "shu", "shou", "enta", "ana", "huwe", "fi", "ma",
    "ya3ne", "tab", "yalla", "lek", "ya3ni"
}
stop_words = english_stop.union(leb_stopwords)

lemmatizer = WordNetLemmatizer()

# ---------------- SLANG MAP ----------------
slang_map = {
    # Weed
    "widad": "weed", "weedad": "weed", "widat": "weed",
    "weed": "weed", "hash": "hash", "hashish": "hash",
    "joint": "joint", "j": "joint",

    # Pills
    "hbboub": "pills", "7abba": "pills", "pill": "pills",
    "pills": "pills", "capsule": "pills", "caps": "pills",
    "dose": "pills", "xans": "pills", "xanax": "pills",
    "valium": "pills", "vals": "pills",
    "hboub": "pills", "habbe": "pills", "7abbe": "pills",

    # Cocaine
    "coke": "cocaine", "snow": "cocaine", "white": "cocaine",
    "powder": "cocaine", "poudre": "cocaine",

    # Weapons
    "gun": "weapon", "sle7": "weapon", "msela7": "weapon",
    "piece": "weapon", "knife": "weapon", "sekkin": "weapon",
    "sekinak": "weapon","sekkinak": "weapon",
}

# -------------- FUZZY MATCH --------------
def fuzzy_match(token: str, options: list, threshold=85):
    for opt in options:
        if fuzz.ratio(token, opt) >= threshold:
            return opt
    return None


# ---------------- NORMALIZATION ----------------
def normalize_variants(token):
    # Regex normalization for weapons
    if re.match(r"sle7\w*", token):
        return "weapon"

    # Cocaine fuzzy
    cocaine_words = ["bayda", "abyad", "coke", "white"]
    match = fuzzy_match(token, cocaine_words)
    if match and match not in ["bayda", "abyad"]:
        return "cocaine"

    # Pills fuzzy
    pills_words = ["pill", "pills", "xans", "xanax", "valium"]
    match = fuzzy_match(token, pills_words)
    if match:
        return "pills"

    return slang_map.get(token, token)


def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [normalize_variants(t) for t in tokens]

    tagged = nltk.pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]

    return " ".join(lemmatized)


class_names = {
    0: "NORMAL",
    1: "WEED SLANG",
    2: "PILLS SLANG",
    3: "COCAINE SLANG",
    4: "WEAPONS SLANG"
}


# ---------------- LOAD OR TRAIN MODEL ----------------
def train_model():
    print("[MODEL] Training new model...")

    df = pd.read_csv(DATASET_PATH)
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )),
        ("clf", LogisticRegression(
            max_iter=800,
            solver="lbfgs",
            multi_class="multinomial"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print("[MODEL] Saved to", MODEL_PATH)


if __name__ == "__main__":
    train_model()
