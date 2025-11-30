from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

# ================== FASTAPI INIT ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODEL ==================
MODEL_PATH = "enhanced_multiclass_model.joblib"
model = joblib.load(MODEL_PATH)

# ================== NLTK SETUP ==================
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


# ======================================================
# ================== PHRASE RULES ======================
# ======================================================
def phrase_rules(text: str):
    t = text.lower().strip()
    clean_t = re.sub(r"\s+", " ", t)
    tokens = clean_t.split()

    # =====================================================
    # 0) WEAPONS — EVERYTHING IS WEAPONS EXCEPT "sekkin matbakh"
    # =====================================================
    weapon_words = [
    # knives
    "sekkin", "sekin", "sekkinak", "sekinak",
    "knife","sekkineh",

    # guns
    "gun", "pistol", "beretta", "glock", "rifle",
    "m16", "ak47", "ak-47", "kalash", "kalashnikov",
    "sniper", "revolver", "magnum",

    # arabic firearms
    "sle7ak", "sle7", "slehak", "sleh", "baroud", "baroude", "baroudi",

    # explosives
    "bomb", "grenade", "tnt", "c4", "explosive", "eneble"
]

    # Safe kitchen knife phrase
    if "sekkin matbakh" in clean_t or "sekin matbakh" in clean_t:
        return "NORMAL"

    # If any weapon term appears → WEAPONS SLANG (no need for violence)
    if any(w in clean_t for w in weapon_words):
        return "WEAPONS SLANG"

    # =====================================================
    # 1) WEED — J slang detection (STRICT & SAFE)
    # =====================================================

    if re.match(r"^j['\s][a-z]", clean_t):
        return "NORMAL"

    if clean_t == "j":
        return "NORMAL"

    if "j" in tokens:
        weed_ctx = ["oerout", "wra2", "wara2", "dakhin", "smoke", "baf", "baff", "joint", "weed"]

        if "chi" in tokens or any(ctx in tokens for ctx in weed_ctx):
            return "WEED SLANG"

        for i in range(len(tokens) - 1):
            if tokens[i] == "bade" and tokens[i + 1] == "j":
                return "WEED SLANG"

    # =====================================================
    # 2) SHORT “bade ...” → NORMAL
    # =====================================================

    if tokens and tokens[0] == "bade" and len(tokens) <= 2:
        suspicious_terms = ["j", "otaata", "abyad", "bayda",
                            "coke", "weed", "hash", "poudre",
                            "snow", "xanax", "xans", "pill", "pills"]
        if all(tok not in suspicious_terms for tok in tokens[1:]):
            return "NORMAL"

    # =====================================================
    # 3) SAFE OBJECTS WITH WHITE → NORMAL
    # =====================================================

    safe_objects = [
        "jeep", "siara", "seyara", "car", "motor", "moteur",
        "kamis", "shirt", "amis", "tshirt", "t-shirt", "hoodie",
        "kanze", "chal7a", "jacket", "pants", "skirt", "dress",
        "shanta", "bag", "purse", "handbag",
        "ghata", "manta", "sofa", "ferse", "tanjara", "banet"
    ]

    for obj in safe_objects:
        if obj in clean_t and ("bayda" in clean_t or "abyad" in clean_t):
            return "NORMAL"

    # =====================================================
    # 4) ARREST TERMS + WHITE → COCAINE
    # =====================================================

    arrest_terms = [
        "kamacho", "kamachou", "kamachoune",
        "kamchou", "kamchoune",
        "masakou", "msaknou", "msaknoune",
        "kebsne", "kabasna", "kebsouni", "kabsouni"
    ]

    if any(a in clean_t for a in arrest_terms) and ("abyad" in clean_t or "bayda" in clean_t):
        return "COCAINE SLANG"

    # =====================================================
    # 5) FOOD CONTEXT → NORMAL
    # =====================================================

    food_verbs = ["ekoul", "akol", "akoul", "bekoul", "bikol",
                  "tekol", "takol", "akalet", "akal", "akalt"]
    food_nouns = ["bayda", "egg", "eggs", "baydetain"]

    if any(v in clean_t for v in food_verbs) and any(n in clean_t for n in food_nouns):
        return "NORMAL"

    # =====================================================
    # 6) HIGH PRIORITY COCAINE PACKET TERMS
    # =====================================================

    cocaine_packet_terms = [
        "otaata el abyad", "otaata abyad", "otaata bayda",
        "otet abyad", "otet bayda",
        "packet abyad", "packet bayda",
        "bag abyad", "bag bayda",
        "l2met abyad", "l2met bayda"
    ]

    if any(term in clean_t for term in cocaine_packet_terms):
        return "COCAINE SLANG"

    # =====================================================
    # 7) DIRECT REQUEST FOR COCAINE
    # =====================================================

    request_verbs = ["bade", "baddi", "bedde",
                     "jib", "jeeb", "jible",
                     "dabbir", "ndabbir"]

    if any(v in clean_t for v in request_verbs) and ("abyad" in clean_t or "bayda" in clean_t):
        return "COCAINE SLANG"

    # =====================================================
    # 8) FALLBACK: WHITE WITHOUT DRUG → NORMAL
    # =====================================================

    if "abyad" in clean_t or "bayda" in clean_t:
        return "NORMAL"

    return None



# ======================================================
# ================== SLANG NORMALIZATION ===============
# ======================================================
slang_map = {
    # Weed
    "widad": "weed", "weedad": "weed", "widat": "weed",
    "weed": "weed", "hash": "hash", "hashish": "hash",
    "joint": "joint",

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
    "gun": "weapon", "sle7": "weapon", "msala7": "weapon","msalla7": "weapon",
      "knife": "weapon", "sekkin": "weapon",
"sekin": "weapon", "sekinak": "weapon", "sekkinak": "weapon", "baroud": "weapon",
"baroude": "weapon", "m16": "weapon", "ak47": "weapon",
"ak-47": "weapon", "kalash": "weapon", "kalashnikov": "weapon",
"pistol": "weapon", "glock": "weapon", "beretta": "weapon",
 "revolver": "weapon", "magnum": "weapon",
"bomb": "weapon", "grenade": "weapon", "tnt": "weapon", "c4": "weapon"
}


def fuzzy_match(token: str, options: list, threshold=85):
    for opt in options:
        if fuzz.ratio(token, opt) >= threshold:
            return opt
    return None


def normalize_variants(token):
    if re.match(r"sle7\w*", token):
        return "weapon"

    cocaine_words = ["bayda", "abyad", "coke", "white"]
    match = fuzzy_match(token, cocaine_words, 80)
    if match:
        if match in ["bayda", "abyad"]:
            return token
        return "cocaine"

    pills_words = ["xans", "xanax", "valium", "pill", "caps"]
    match = fuzzy_match(token, pills_words, 80)
    if match:
        return "pills"

    return slang_map.get(token, token)


def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
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


# ================== API ENDPOINT ==================
@app.post("/analyze")
def analyze_text(data: dict):
    text = data.get("text", "")

    if not text.strip():
        return {"label": "NORMAL", "confidence": 0.0}

    # Phrase-level logic first
    rule = phrase_rules(text)
    if rule:
        return {
            "original": text,
            "cleaned": text.lower(),
            "label": rule,
            "confidence": 1.0
        }

    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    predicted_index = probs.argmax()

    return {
        "original": text,
        "cleaned": cleaned,
        "label": class_names[predicted_index],
        "confidence": float(probs[predicted_index])
    }
