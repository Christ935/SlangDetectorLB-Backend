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

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

# ---------------- SLANG NORMALIZATION ----------------

slang_map = {
    "widad": "weed", "weedad": "weed", "widat": "weed",
    "weed": "weed", "hash": "hash", "hashish": "hash",
    "joint": "joint", "j": "joint",

    "hbboub": "pills", "7abba": "pills", "pill": "pills",
    "pills": "pills", "capsule": "pills", "caps": "pills",
    "dose": "pills", "xans": "pills", "xanax": "pills",
    "valium": "pills", "vals": "pills", "blue": "pills",
    "hboub": "pills", "habbe": "pills", "7abbe": "pills",

    "coke": "cocaine", "snow": "cocaine", "white": "cocaine",
    "abyad": "cocaine", "bayda": "cocaine",

    "gun": "weapon", "sle7": "weapon", "msela7": "weapon",
    "piece": "weapon", "knife": "weapon", "sekkin": "weapon",
}

def normalize_variants(token):
    if re.match(r"sle7\w*", token): return "weapon"
    if re.match(r"xans\w*", token): return "pills"
    if re.match(r"abyad\w*", token): return "cocaine"
    return slang_map.get(token, token)

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

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("[MODEL] Loading existing model...")
        return joblib.load(MODEL_PATH)

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
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )),
        ("clf", LogisticRegression(
            max_iter=400,
            solver="lbfgs",
            multi_class="multinomial"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    return model


model = load_or_train_model()

# ---------------- API FUNCTION ----------------

def predict_for_api(text: str):
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    predicted_class = int(probs.argmax())
    confidence = float(probs[predicted_class])

    return {
        "original": text,
        "cleaned": cleaned,
        "label": class_names[predicted_class],
        "class_id": predicted_class,
        "confidence": confidence
    }
