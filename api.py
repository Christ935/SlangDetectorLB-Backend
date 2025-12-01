from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
import logging

# ================== FASTAPI INIT ==================
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

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
    "ya","wallah","shu","shou","enta","ana","huwe","fi","ma",
    "ya3ne","tab","yalla","lek","ya3ni","w","eh","ahu","akid",
    "fiya","elha","hala2","hayda","hayde","haydi","hot"
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
    # 4) COCAINE RULES (NEW FIX)
    # =====================================================

    text_words = clean_t.split()

    # --- 4.1 HIGH PRIORITY: "el abyad" / "el bayda" ---
    if "el abyad" in clean_t or "el bayda" in clean_t:
        return "COCAINE SLANG"

    # --- 4.2 "abyad" ALWAYS = cocaine ---
    if "abyad" in clean_t:
        return "COCAINE SLANG"

    # --- 4.3 BAYDA ALONE (SAFE UNLESS DRUG CONTEXT) ---
    cocaine_context = [
        "otaata", "packet", "poudra", "powder",
        "l2met", "bag", "kabsna", "kamacho"
    ]

    if "bayda" in clean_t:
        # if any cocaine context appears → cocaine slang
        if any(c in clean_t for c in cocaine_context):
            return "COCAINE SLANG"
        # else → NORMAL (egg, shirt bayda, etc)
        # will be returned by later logic
        pass

    # =====================================================
    # 4.4 ARREST TERMS + white (still work)
    # =====================================================

    arrest_terms = [
        "kamacho", "kamachou", "kamachoune",
        "kamchou", "kamchoune",
        "masakou", "msaknou", "msaknoune",
        "kebsne", "kabasna", "kebsouni", "kabsouni"
    ]

    if any(a in clean_t for a in arrest_terms) and ("abyad" in clean_t or "el abyad" in clean_t):
        return "COCAINE SLANG"

    # =====================================================
    # 4.5 COCAINE packet terms with bayda
    # =====================================================

    cocaine_packet_terms = [
        "otaata bayda", "otet bayda",
        "packet bayda", "bag bayda",
        "l2met bayda", "poudra bayda"
    ]

    if any(term in clean_t for term in cocaine_packet_terms):
        return "COCAINE SLANG"

    # =====================================================
    # 4.6 DIRECT REQUEST FOR COCAINE
    # =====================================================

    request_verbs = [
        "bade", "baddi", "bedde",
        "jebet", "jib", "jeeb", "jible",
        "ma3ak", "3andak",
        "dabbir", "ndabbir"
    ]

    if "abyad" in clean_t or "el abyad" in clean_t or "el bayda" in clean_t:
        if any(v in clean_t for v in request_verbs):
            return "COCAINE SLANG"

    # if only "bayda" + request verb → still NORMAL unless cocaine context
    if "bayda" in clean_t and any(v in clean_t for v in request_verbs):
        if any(c in clean_t for c in cocaine_context):
            return "COCAINE SLANG"
        return "NORMAL"


    # =====================================================
    # 8) FALLBACK: WHITE WITHOUT DRUG → NORMAL
    # =====================================================
    pills_keywords = ["xans", "xan", "xanax", "hboub", "habbe", "7abbe", "pills", "pill"]
    pills_request_verbs = [
        "jebet", "jib", "jeeb", "jible", "3andak", "ma3ak",
        "bade", "baddi", "bedde",
        "btejeeb", "btjeeb", "dabbir", "ndabbir", "dabbirlé"
    ]

    if any(pk in clean_t for pk in pills_keywords) and any(rv in clean_t for rv in pills_request_verbs):
        return "PILLS SLANG"

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
    logger.info("\n================ CLEANING PIPELINE ================")
    logger.info(f"RAW INPUT: {text}")

    # 1) lower + punctuation
    lowered = str(text).lower()
    cleaned_punc = re.sub(r"[^\w\s]", " ", lowered)
    logger.info(f"LOWERCASE & NO PUNCT: {cleaned_punc}")

    # 2) tokenize
    tokens = word_tokenize(cleaned_punc)
    logger.info(f"TOKENS: {tokens}")

    # 3) remove stopwords
    tokens_no_stop = [t for t in tokens if t not in stop_words]
    logger.info(f"AFTER STOPWORD REMOVAL: {tokens_no_stop}")

    # 4) slang normalization
    normalized = [normalize_variants(t) for t in tokens_no_stop]
    logger.info(f"AFTER SLANG NORMALIZATION: {normalized}")

    # 5) POS tagging
    tagged = nltk.pos_tag(normalized)
    logger.info(f"POS TAGS: {tagged}")

    # 6) lemmatization
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    logger.info(f"LEMMATIZED TOKENS: {lemmatized}")

    # 7) final string
    final_cleaned = " ".join(lemmatized)
    logger.info(f"FINAL CLEANED TEXT: {final_cleaned}")
    logger.info("====================================================\n")

    return final_cleaned






# ================== API ENDPOINT ==================
@app.post("/analyze")
def analyze_text(data: dict):
    text = data.get("text", "")

    if not text.strip():
        return {"label": "NORMAL", "confidence": 0.0}

    # ==========================================================
    # 1) ALWAYS CLEAN FIRST
    # ==========================================================
    cleaned = clean_text(text)

    # ==========================================================
    # 2) MODEL PREDICTION
    # ==========================================================
    probs = model.predict_proba([cleaned])[0]
    model_idx = probs.argmax()
    model_label = class_names[model_idx]
    model_conf = float(probs[model_idx])

    # ==========================================================
    # 3) RULE-BASED CLASSIFICATION (RAW AND CLEANED)
    # ==========================================================
    rule_label_raw = phrase_rules(text)
    rule_label_cleaned = phrase_rules(cleaned)

    # Pick strongest rule result
    rule_label = rule_label_raw or rule_label_cleaned

        # ==========================================================
    # 4) MERGE RULE + MODEL WITH 60% SUSPICIOUS THRESHOLD
    # ==========================================================

    THRESHOLD = 0.60   # ← only suspicious if model ≥ 60%

    # --- 4.1 WEAPONS ALWAYS OVERRIDE ---
    if rule_label == "WEAPONS SLANG":
        return {
            "original": text,
            "cleaned": cleaned,
            "label": "WEAPONS SLANG",
            "confidence": 1.0,
            "source": "RULE_STRONG"
        }

    # --- 4.2 If RULE detects slang → TRUST RULE ---
    if rule_label and rule_label != "NORMAL":
        return {
            "original": text,
            "cleaned": cleaned,
            "label": rule_label,
            "confidence": 1.0,
            "source": "RULE_OVERRIDE"
        }

    # --- 4.3 MODEL detects slang BUT BELOW THRESHOLD → NORMAL ---
    if model_label != "NORMAL" and model_conf < THRESHOLD:
        return {
            "original": text,
            "cleaned": cleaned,
            "label": "NORMAL",
            "confidence": model_conf,
            "source": "MODEL_BELOW_THRESHOLD"
        }

    # --- 4.4 MODEL detects slang AND ABOVE THRESHOLD → use it ---
    if model_label != "NORMAL" and model_conf >= THRESHOLD:
        return {
            "original": text,
            "cleaned": cleaned,
            "label": model_label,
            "confidence": model_conf,
            "source": "MODEL_STRONG"
        }

    # --- 4.5 NOTHING SUSPICIOUS (rules NORMAL + model NORMAL) ---
    return {
        "original": text,
        "cleaned": cleaned,
        "label": "NORMAL",
        "confidence": model_conf,
        "source": "CLEAN"
    }
