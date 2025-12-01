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

# First we set up logging and the FastAPI app 

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Set up CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model

MODEL_PATH = "enhanced_multiclass_model.joblib"
model = joblib.load(MODEL_PATH)


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# We start by defining the stop words which include both English and Lebanese Arabic slang stop words that are common words that do not contribute much to the meaning of the text.

english_stop = set(stopwords.words("english"))
leb_stopwords = {
    "ya","wallah","shu","shou","enta","ana","huwe","fi","ma",
    "ya3ne","tab","yalla","lek","ya3ni","w","eh","ahu","akid",
    "fiya","elha","hala2","hayda","hayde","haydi","hot"
}

stop_words = english_stop.union(leb_stopwords)

# Initialize the lemmatizer to reduce words to their base or root form
lemmatizer = WordNetLemmatizer()

# Define class names for the different categories
class_names = {
    0: "NORMAL",
    1: "WEED SLANG",
    2: "PILLS SLANG",
    3: "COCAINE SLANG",
    4: "WEAPONS SLANG"
}


# In this function, we apply specific phrase-based rules to classify the text into different categories and help our model make more accurate predictions.

def phrase_rules(text: str):
    #We first normalize the text by converting it to lowercase and removing extra spaces
    t = text.lower().strip()
    clean_t = re.sub(r"\s+", " ", t)
    tokens = clean_t.split()

   # Define a list of weapon-related words to identify weapons slang
    weapon_words = [
    
    "sekkin", "sekin", "sekkinak", "sekinak",
    "knife","sekkineh",

    
    "gun", "pistol", "beretta", "glock", "rifle",
    "m16", "ak47", "ak-47", "kalash", "kalashnikov",
    "sniper", "revolver", "magnum",

    
    "sle7ak", "sle7", "slehak", "sleh", "baroud", "baroude", "baroudi",

    
    "bomb", "grenade", "tnt", "c4", "explosive", "eneble"
]

    # Check for specific phrase that should be classified as NORMAL despite containing weapon words
    if "sekkin matbakh" in clean_t or "sekin matbakh" in clean_t:
        return "NORMAL"

    
    # If any weapon-related word is found in the text, classify it as WEAPONS SLANG
    if any(w in clean_t for w in weapon_words):
        return "WEAPONS SLANG"

    
    # Handle cases related to the slang term "j" which can refer to a joint (weed) or be used in normal contexts
    if re.match(r"^j['\s][a-z]", clean_t):
        return "NORMAL"
    
    # Handle the case where it could be a typo
    if clean_t == "j":
        return "NORMAL"
    
    # Check if the slang term "j" is used in a context related to weed
    if "j" in tokens:
        weed_ctx = ["oerout", "wra2", "wara2", "dakhin", "smoke", "baf", "baff", "joint", "weed"]

        if "chi" in tokens or any(ctx in tokens for ctx in weed_ctx):
            return "WEED SLANG"

        for i in range(len(tokens) - 1):
            if tokens[i] == "bade" and tokens[i + 1] == "j":
                return "WEED SLANG"

    # Specific rule for "bade" (I want) followed by non-suspicious terms

    if tokens and tokens[0] == "bade" and len(tokens) <= 2:
        suspicious_terms = ["j", "otaata", "abyad", "bayda",
                            "coke", "weed", "hash", "poudre",
                            "snow", "xanax", "xans", "pill", "pills"]
        if all(tok not in suspicious_terms for tok in tokens[1:]):
            return "NORMAL"

    # Define a list of safe objects that, when mentioned alongside suspicious terms, indicate a NORMAL context

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

        

    text_words = clean_t.split()

    
    if "el abyad" in clean_t or "el bayda" in clean_t:
        return "COCAINE SLANG"

    
    if "abyad" in clean_t:
        return "COCAINE SLANG"

    
    cocaine_context = [
        "otaata", "packet", "poudra", "powder",
        "l2met", "bag", "kabsna", "kamacho"
    ]

    if "bayda" in clean_t:
        
        if any(c in clean_t for c in cocaine_context):
            return "COCAINE SLANG"
        
        pass

   

    arrest_terms = [
        "kamacho", "kamachou", "kamachoune",
        "kamchou", "kamchoune",
        "masakou", "msaknou", "msaknoune",
        "kebsne", "kabasna", "kebsouni", "kabsouni"
    ]

    if any(a in clean_t for a in arrest_terms) and ("abyad" in clean_t or "el abyad" in clean_t):
        return "COCAINE SLANG"



    cocaine_packet_terms = [
        "otaata bayda", "otet bayda",
        "packet bayda", "bag bayda",
        "l2met bayda", "poudra bayda"
    ]

    if any(term in clean_t for term in cocaine_packet_terms):
        return "COCAINE SLANG"

  

    request_verbs = [
        "bade", "baddi", "bedde",
        "jebet", "jib", "jeeb", "jible",
        "ma3ak", "3andak",
        "dabbir", "ndabbir"
    ]

    if "abyad" in clean_t or "el abyad" in clean_t or "el bayda" in clean_t:
        if any(v in clean_t for v in request_verbs):
            return "COCAINE SLANG"

    
    if "bayda" in clean_t and any(v in clean_t for v in request_verbs):
        if any(c in clean_t for c in cocaine_context):
            return "COCAINE SLANG"
        return "NORMAL"


   
    pills_keywords = ["xans", "xan", "xanax", "hboub", "habbe", "7abbe", "pills", "pill"]
    pills_request_verbs = [
        "jebet", "jib", "jeeb", "jible", "3andak", "ma3ak",
        "bade", "baddi", "bedde",
        "btejeeb", "btjeeb", "dabbir", "ndabbir", "dabbirlé","dabberle"
    ]

    if any(pk in clean_t for pk in pills_keywords) and any(rv in clean_t for rv in pills_request_verbs):
        return "PILLS SLANG"

    if "abyad" in clean_t or "bayda" in clean_t:
        return "NORMAL"

    return None



# We define a mapping of slang variants to their normalized forms to standardize different spellings and variations of slang terms.
slang_map = {
    
    "widad": "weed", "weedad": "weed", "widat": "weed",
    "weed": "weed", "hash": "hash", "hashish": "hash",
    "joint": "joint", "7achich": "hash", "j": "joint",

    
    "hbboub": "pills", "7abba": "pills", "pill": "pills",
    "pills": "pills", "capsule": "pills", "caps": "pills",
    "dose": "pills", "xans": "pills", "xanax": "pills",
    "valium": "pills", "vals": "pills",
    "hboub": "pills", "habbe": "pills", "7abbe": "pills",

    
    "coke": "cocaine", "snow": "cocaine", "white": "cocaine",
    "powder": "cocaine", "poudre": "cocaine",

    
    "gun": "weapon", "sle7": "weapon", "msala7": "weapon","msalla7": "weapon",
      "knife": "weapon", "sekkin": "weapon",
"sekin": "weapon", "sekinak": "weapon", "sekkinak": "weapon", "baroud": "weapon",
"baroude": "weapon", "m16": "weapon", "ak47": "weapon",
"ak-47": "weapon", "kalash": "weapon", "kalashnikov": "weapon",
"pistol": "weapon", "glock": "weapon", "beretta": "weapon",
 "revolver": "weapon", "magnum": "weapon",
"bomb": "weapon", "grenade": "weapon", "tnt": "weapon", "c4": "weapon"
}


# Using fuzzy matching to account for typos and variations in slang terms

def fuzzy_match(token: str, options: list, threshold=85):
    for opt in options:
        if fuzz.ratio(token, opt) >= threshold:
            return opt
    return None

# Normalize slang variants using regex and fuzzy matching
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


#POS is used for Part Of Speech tagging that means identifying whether a word is a noun, verb, adjective, or adverb.
def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

# We re clean the text by applying several preprocessing steps such as lowercasing, removing punctuation, tokenization, stop word removal, slang normalization, POS tagging, and lemmatization. and
def clean_text(text):
    logger.info("\n================ CLEANING PIPELINE ================")
    logger.info(f"RAW INPUT: {text}")

    
    lowered = str(text).lower()
    cleaned_punc = re.sub(r"[^\w\s]", " ", lowered)
    logger.info(f"LOWERCASE & NO PUNCT: {cleaned_punc}")

   
    tokens = word_tokenize(cleaned_punc)
    logger.info(f"TOKENS: {tokens}")

    
    tokens_no_stop = [t for t in tokens if t not in stop_words]
    logger.info(f"AFTER STOPWORD REMOVAL: {tokens_no_stop}")

    
    normalized = [normalize_variants(t) for t in tokens_no_stop]
    logger.info(f"AFTER SLANG NORMALIZATION: {normalized}")

    
    tagged = nltk.pos_tag(normalized)
    logger.info(f"POS TAGS: {tagged}")

    
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    logger.info(f"LEMMATIZED TOKENS: {lemmatized}")

    
    final_cleaned = " ".join(lemmatized)
    logger.info(f"FINAL CLEANED TEXT: {final_cleaned}")
    logger.info("====================================================\n")

    return final_cleaned








# We define the main API endpoint /analyze that accepts POST requests with text data to analyze and classify it using both rule-based and model-based approaches.

@app.post("/analyze")
def analyze_text(data: dict):
    text = data.get("text", "")

    if not text.strip():
        return {"label": "NORMAL", "confidence": 0.0}

    
    cleaned = clean_text(text)

    
    probs = model.predict_proba([cleaned])[0]
    model_idx = probs.argmax()
    model_label = class_names[model_idx]
    model_conf = float(probs[model_idx])

    
    rule_label_raw = phrase_rules(text)
    rule_label_cleaned = phrase_rules(cleaned)

    
    rule_label = rule_label_raw or rule_label_cleaned

    
    THRESHOLD = 0.60   

    
    if rule_label == "WEAPONS SLANG":
        return {
            "original": text,
            "cleaned": cleaned,
            "label": "WEAPONS SLANG",
            "confidence": 1.0,
            "source": "RULE_STRONG"
        }

    
    if rule_label and rule_label != "NORMAL":
        return {
            "original": text,
            "cleaned": cleaned,
            "label": rule_label,
            "confidence": 1.0,
            "source": "RULE_OVERRIDE"
        }

    
    if model_label != "NORMAL" and model_conf < THRESHOLD:
        return {
            "original": text,
            "cleaned": cleaned,
            "label": "NORMAL",
            "confidence": model_conf,
            "source": "MODEL_BELOW_THRESHOLD"
        }

    
    if model_label != "NORMAL" and model_conf >= THRESHOLD:
        return {
            "original": text,
            "cleaned": cleaned,
            "label": model_label,
            "confidence": model_conf,
            "source": "MODEL_STRONG"
        }

    
    return {
        "original": text,
        "cleaned": cleaned,
        "label": "NORMAL",
        "confidence": model_conf,
        "source": "CLEAN"
    }
