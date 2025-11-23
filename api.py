from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ================== FASTAPI INIT ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODEL ==================
MODEL_PATH = "enhanced_multiclass_model.joblib"
model = joblib.load(MODEL_PATH)

# ================== NLTK ==================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

english_stop = set(stopwords.words("english"))
leb_stopwords = {
    "ya", "wallah", "shu", "shou", "enta", "ana", "huwe", "fi", "ma",
    "ya3ne", "tab", "yalla", "lek", "ya3ni"
}
stop_words = english_stop.union(leb_stopwords)

lemmatizer = WordNetLemmatizer()

# ================== LABELS ==================
class_names = {
    0: "NORMAL",
    1: "WEED SLANG",
    2: "PILLS SLANG",
    3: "COCAINE SLANG",
    4: "WEAPONS SLANG"
}

# ================== SLANG MAP (IDENTICAL) ==================
slang_map = {
    # Weed
    "widad": "weed", "weedad": "weed", "widat": "weed",
    "weed": "weed", "hash": "hash", "hashish": "hash",
    "joint": "joint", "j": "joint",

    # Pills
    "hbboub": "pills", "7abba": "pills", "pill": "pills",
    "pills": "pills", "capsule": "pills", "caps": "pills",
    "dose": "pills", "xans": "pills", "xanax": "pills",
    "valium": "pills", "vals": "pills", "blue": "pills",
    "hboub": "pills", "habbe": "pills", "7abbe": "pills",
    "habbit": "pills", "7abbit": "pills",

    # Cocaine
    "coke": "cocaine", "snow": "cocaine", "white": "cocaine",
    "abyad": "cocaine", "bayda": "cocaine", "elbayda": "cocaine",

    # Weapons
    "gun": "weapon", "sle7": "weapon", "slehak": "weapon",
    "sle7ak": "weapon", "msela7": "weapon", "msalla7": "weapon",
    "piece": "weapon", "knife": "weapon", "sikkeen": "weapon",
    "sekkin": "weapon", "sekin": "weapon", "sekkinak": "weapon",
    "sekinak": "weapon",
}

# ================== HELPERS ==================
def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN


def normalize_variants(token):
    if re.match(r"sle7\w*", token) or re.match(r"msela7\w*", token):
        return "weapon"
    if re.match(r"xans\w*", token) or re.match(r"xanax\w*", token) or re.match(r"7abba\w*", token):
        return "pills"
    if re.match(r"pill\w*", token) or re.match(r"caps\w*", token):
        return "pills"
    if re.match(r"abyad\w*", token) or re.match(r"bayda\w*", token):
        return "cocaine"
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


# ================== API ENDPOINT ==================
@app.post("/analyze")
def analyze_text(data: dict):

    text = data.get("text", "")

    if not text.strip():
        return {"label": "NORMAL", "confidence": 0.0}

    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    predicted_index = probs.argmax()

    return {
        "original": text,
        "cleaned": cleaned,
        "label": class_names[predicted_index],
        "confidence": float(probs[predicted_index])
    }
