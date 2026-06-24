"""
Tamil GEC Training Data Generator for TaGI Framework
Generates data/train.jsonl and data/test.jsonl with annotated Tamil sentences.
Error types: ME(1), TE(2), SVA(3), CME(4), SV(5), WOE(6), NAE(7), SPE(8)
Run: python data/generate_data.py
"""
import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)
Path("data").mkdir(exist_ok=True)

# ─── Vocabulary (token → integer ID) ─────────────────────────────────────────
VOCAB = {
    "நான்":1,"நாங்கள்":2,"நீ":3,"நீங்கள்":4,
    "அவன்":5,"அவள்":6,"அவர்":7,"அவர்கள்":8,"அது":9,
    "பள்ளி":10,"வீடு":11,"வகுப்பு":12,"சந்தை":13,"மருத்துவமனை":14,
    "நூலகம்":15,"கோவில்":16,"வயல்":17,"நகரம்":18,"காடு":19,
    "மாணவன்":20,"மாணவள்":21,"மாணவர்கள்":22,"மாணவர்":23,
    "ஆசிரியர்":25,"ஆசிரியர்கள்":26,"மருத்துவர்":27,"இயக்குனர்":28,
    "புத்தகம்":30,"கடிதம்":31,"பேனா":32,"பை":33,"கணினி":34,
    "சாப்பாடு":35,"தண்ணீர்":36,"பழம்":37,"காய்கறி":38,"பால்":39,
    "அம்மா":40,"அப்பா":41,"அண்ணன்":42,"தங்கை":43,"அக்கா":44,
    "நண்பன்":45,"நண்பர்கள்":46,"குழந்தை":47,"குழந்தைகள்":48,
    # Locative (LOC) case forms
    "பள்ளியில்":50,"வீட்டில்":51,"வகுப்பில்":52,"சந்தையில்":53,
    "மருத்துவமனையில்":54,"நூலகத்தில்":55,"கோவிலில்":56,"வயலில்":57,
    "நகரத்தில்":58,"காட்டில்":59,
    # Dative (DAT) case forms
    "பள்ளிக்கு":60,"வீட்டிற்கு":61,"சந்தைக்கு":62,"மருத்துவமனைக்கு":63,
    "அம்மாவிற்கு":64,"அப்பாவிற்கு":65,"நண்பனுக்கு":66,"ஆசிரியருக்கு":67,
    "நூலகத்திற்கு":68,"கோவிலுக்கு":69,
    # Accusative (ACC) case forms
    "புத்தகத்தை":70,"கடிதத்தை":71,"சாப்பாட்டை":72,
    "தண்ணீரை":73,"பழத்தை":74,"பையை":75,"கணினியை":76,"பாலை":77,
    # Adverbs
    "தினமும்":80,"இன்று":81,"நேற்று":82,"நாளை":83,
    "விரைவாக":84,"மெதுவாக":85,"நன்றாக":86,
    "கவனமாக":87,"எப்போதும்":88,"அடிக்கடி":89,
    # Present tense verbs  (100–198)
    "படிக்கிறேன்":100,"படிக்கிறோம்":101,"படிக்கிறாய்":102,"படிக்கிறீர்கள்":103,
    "படிக்கிறான்":104,"படிக்கிறாள்":105,"படிக்கிறார்":106,"படிக்கிறது":107,"படிக்கிறார்கள்":108,
    "போகிறேன்":110,"போகிறோம்":111,"போகிறாய்":112,"போகிறீர்கள்":113,
    "போகிறான்":114,"போகிறாள்":115,"போகிறார்":116,"போகிறது":117,"போகிறார்கள்":118,
    "வருகிறேன்":120,"வருகிறோம்":121,"வருகிறாய்":122,"வருகிறீர்கள்":123,
    "வருகிறான்":124,"வருகிறாள்":125,"வருகிறார்":126,"வருகிறது":127,"வருகிறார்கள்":128,
    "சாப்பிடுகிறேன்":130,"சாப்பிடுகிறோம்":131,"சாப்பிடுகிறாய்":132,"சாப்பிடுகிறீர்கள்":133,
    "சாப்பிடுகிறான்":134,"சாப்பிடுகிறாள்":135,"சாப்பிடுகிறார்":136,
    "சாப்பிடுகிறது":137,"சாப்பிடுகிறார்கள்":138,
    "எழுதுகிறேன்":140,"எழுதுகிறோம்":141,"எழுதுகிறாய்":142,"எழுதுகிறீர்கள்":143,
    "எழுதுகிறான்":144,"எழுதுகிறாள்":145,"எழுதுகிறார்":146,
    "எழுதுகிறது":147,"எழுதுகிறார்கள்":148,
    "பார்க்கிறேன்":150,"பார்க்கிறோம்":151,"பார்க்கிறாய்":152,"பார்க்கிறீர்கள்":153,
    "பார்க்கிறான்":154,"பார்க்கிறாள்":155,"பார்க்கிறார்":156,
    "பார்க்கிறது":157,"பார்க்கிறார்கள்":158,
    "ஓடுகிறேன்":160,"ஓடுகிறோம்":161,"ஓடுகிறான்":164,"ஓடுகிறாள்":165,
    "ஓடுகிறார்":166,"ஓடுகிறார்கள்":168,
    "பேசுகிறேன்":170,"பேசுகிறோம்":171,"பேசுகிறான்":174,"பேசுகிறாள்":175,
    "பேசுகிறார்":176,"பேசுகிறார்கள்":178,
    "விளையாடுகிறேன்":180,"விளையாடுகிறோம்":181,"விளையாடுகிறான்":184,
    "விளையாடுகிறாள்":185,"விளையாடுகிறார்":186,"விளையாடுகிறார்கள்":188,
    "இருக்கிறேன்":190,"இருக்கிறோம்":191,"இருக்கிறான்":194,"இருக்கிறாள்":195,
    "இருக்கிறார்":196,"இருக்கிறார்கள்":198,
    # Past tense verbs (200–268)
    "படித்தேன்":200,"படித்தோம்":201,"படித்தான்":204,"படித்தாள்":205,
    "படித்தார்":206,"படித்தார்கள்":208,
    "போனேன்":210,"போனோம்":211,"போனான்":214,"போனாள்":215,
    "போனார்":216,"போனார்கள்":218,
    "வந்தேன்":220,"வந்தோம்":221,"வந்தான்":224,"வந்தாள்":225,
    "வந்தார்":226,"வந்தார்கள்":228,
    "சாப்பிட்டேன்":230,"சாப்பிட்டோம்":231,"சாப்பிட்டான்":234,"சாப்பிட்டாள்":235,
    "சாப்பிட்டார்":236,"சாப்பிட்டார்கள்":238,
    "எழுதினேன்":240,"எழுதினோம்":241,"எழுதினான்":244,"எழுதினாள்":245,
    "எழுதினார்":246,"எழுதினார்கள்":248,
    "பார்த்தேன்":250,"பார்த்தோம்":251,"பார்த்தான்":254,"பார்த்தாள்":255,
    "பார்த்தார்":256,"பார்த்தார்கள்":258,
    "ஓடினேன்":260,"ஓடினோம்":261,"ஓடினான்":264,"ஓடினாள்":265,
    "ஓடினார்":266,"ஓடினார்கள்":268,
    # Future tense verbs (300–338)
    "படிப்பேன்":300,"படிப்போம்":301,"படிப்பான்":304,"படிப்பாள்":305,
    "படிப்பார்":306,"படிப்பார்கள்":308,
    "போவேன்":310,"போவோம்":311,"போவான்":314,"போவாள்":315,
    "போவார்":316,"போவார்கள்":318,
    "வருவேன்":320,"வருவோம்":321,"வருவான்":324,"வருவாள்":325,
    "வருவார்":326,"வருவார்கள்":328,
    "சாப்பிடுவேன்":330,"சாப்பிடுவோம்":331,"சாப்பிடுவான்":334,"சாப்பிடுவாள்":335,
    "சாப்பிடுவார்":336,"சாப்பிடுவார்கள்":338,
    # Punctuation
    ".":990,",":991,"!":992,"?":993,
}

# ─── Pronoun features ─────────────────────────────────────────────────────────
PRON_FEAT = {
    "நான்":    ("1","SG",""),  "நாங்கள்": ("1","PL",""),
    "நீ":      ("2","SG",""),  "நீங்கள்": ("2","PL",""),
    "அவன்":    ("3","SG","M"), "அவள்":    ("3","SG","F"),
    "அவர்":    ("3","SG","H"), "அவர்கள்": ("3","PL",""),
    "அது":     ("3","SG","N"),
}

# ─── Verb paradigms ───────────────────────────────────────────────────────────
VERBS = {
    ("படி","present"): {
        ("1","SG",""):"படிக்கிறேன்",("1","PL",""):"படிக்கிறோம்",
        ("2","SG",""):"படிக்கிறாய்",("2","PL",""):"படிக்கிறீர்கள்",
        ("3","SG","M"):"படிக்கிறான்",("3","SG","F"):"படிக்கிறாள்",
        ("3","SG","H"):"படிக்கிறார்",("3","SG","N"):"படிக்கிறது",
        ("3","PL",""):"படிக்கிறார்கள்",
    },
    ("போ","present"): {
        ("1","SG",""):"போகிறேன்",("1","PL",""):"போகிறோம்",
        ("2","SG",""):"போகிறாய்",("2","PL",""):"போகிறீர்கள்",
        ("3","SG","M"):"போகிறான்",("3","SG","F"):"போகிறாள்",
        ("3","SG","H"):"போகிறார்",("3","SG","N"):"போகிறது",
        ("3","PL",""):"போகிறார்கள்",
    },
    ("வா","present"): {
        ("1","SG",""):"வருகிறேன்",("1","PL",""):"வருகிறோம்",
        ("2","SG",""):"வருகிறாய்",("2","PL",""):"வருகிறீர்கள்",
        ("3","SG","M"):"வருகிறான்",("3","SG","F"):"வருகிறாள்",
        ("3","SG","H"):"வருகிறார்",("3","SG","N"):"வருகிறது",
        ("3","PL",""):"வருகிறார்கள்",
    },
    ("சாப்பிடு","present"): {
        ("1","SG",""):"சாப்பிடுகிறேன்",("1","PL",""):"சாப்பிடுகிறோம்",
        ("2","SG",""):"சாப்பிடுகிறாய்",("2","PL",""):"சாப்பிடுகிறீர்கள்",
        ("3","SG","M"):"சாப்பிடுகிறான்",("3","SG","F"):"சாப்பிடுகிறாள்",
        ("3","SG","H"):"சாப்பிடுகிறார்",("3","SG","N"):"சாப்பிடுகிறது",
        ("3","PL",""):"சாப்பிடுகிறார்கள்",
    },
    ("எழுது","present"): {
        ("1","SG",""):"எழுதுகிறேன்",("1","PL",""):"எழுதுகிறோம்",
        ("2","SG",""):"எழுதுகிறாய்",("2","PL",""):"எழுதுகிறீர்கள்",
        ("3","SG","M"):"எழுதுகிறான்",("3","SG","F"):"எழுதுகிறாள்",
        ("3","SG","H"):"எழுதுகிறார்",("3","SG","N"):"எழுதுகிறது",
        ("3","PL",""):"எழுதுகிறார்கள்",
    },
    ("பார்","present"): {
        ("1","SG",""):"பார்க்கிறேன்",("1","PL",""):"பார்க்கிறோம்",
        ("2","SG",""):"பார்க்கிறாய்",("2","PL",""):"பார்க்கிறீர்கள்",
        ("3","SG","M"):"பார்க்கிறான்",("3","SG","F"):"பார்க்கிறாள்",
        ("3","SG","H"):"பார்க்கிறார்",("3","SG","N"):"பார்க்கிறது",
        ("3","PL",""):"பார்க்கிறார்கள்",
    },
    ("ஓடு","present"): {
        ("1","SG",""):"ஓடுகிறேன்",("1","PL",""):"ஓடுகிறோம்",
        ("3","SG","M"):"ஓடுகிறான்",("3","SG","F"):"ஓடுகிறாள்",
        ("3","SG","H"):"ஓடுகிறார்",("3","PL",""):"ஓடுகிறார்கள்",
    },
    ("பேசு","present"): {
        ("1","SG",""):"பேசுகிறேன்",("1","PL",""):"பேசுகிறோம்",
        ("3","SG","M"):"பேசுகிறான்",("3","SG","F"):"பேசுகிறாள்",
        ("3","SG","H"):"பேசுகிறார்",("3","PL",""):"பேசுகிறார்கள்",
    },
    ("விளையாடு","present"): {
        ("1","SG",""):"விளையாடுகிறேன்",("1","PL",""):"விளையாடுகிறோம்",
        ("3","SG","M"):"விளையாடுகிறான்",("3","SG","F"):"விளையாடுகிறாள்",
        ("3","SG","H"):"விளையாடுகிறார்",("3","PL",""):"விளையாடுகிறார்கள்",
    },
    ("இரு","present"): {
        ("1","SG",""):"இருக்கிறேன்",("1","PL",""):"இருக்கிறோம்",
        ("3","SG","M"):"இருக்கிறான்",("3","SG","F"):"இருக்கிறாள்",
        ("3","SG","H"):"இருக்கிறார்",("3","PL",""):"இருக்கிறார்கள்",
    },
    ("படி","past"): {
        ("1","SG",""):"படித்தேன்",("1","PL",""):"படித்தோம்",
        ("3","SG","M"):"படித்தான்",("3","SG","F"):"படித்தாள்",
        ("3","SG","H"):"படித்தார்",("3","PL",""):"படித்தார்கள்",
    },
    ("போ","past"): {
        ("1","SG",""):"போனேன்",("1","PL",""):"போனோம்",
        ("3","SG","M"):"போனான்",("3","SG","F"):"போனாள்",
        ("3","SG","H"):"போனார்",("3","PL",""):"போனார்கள்",
    },
    ("வா","past"): {
        ("1","SG",""):"வந்தேன்",("1","PL",""):"வந்தோம்",
        ("3","SG","M"):"வந்தான்",("3","SG","F"):"வந்தாள்",
        ("3","SG","H"):"வந்தார்",("3","PL",""):"வந்தார்கள்",
    },
    ("சாப்பிடு","past"): {
        ("1","SG",""):"சாப்பிட்டேன்",("1","PL",""):"சாப்பிட்டோம்",
        ("3","SG","M"):"சாப்பிட்டான்",("3","SG","F"):"சாப்பிட்டாள்",
        ("3","SG","H"):"சாப்பிட்டார்",("3","PL",""):"சாப்பிட்டார்கள்",
    },
    ("எழுது","past"): {
        ("1","SG",""):"எழுதினேன்",("1","PL",""):"எழுதினோம்",
        ("3","SG","M"):"எழுதினான்",("3","SG","F"):"எழுதினாள்",
        ("3","SG","H"):"எழுதினார்",("3","PL",""):"எழுதினார்கள்",
    },
    ("பார்","past"): {
        ("1","SG",""):"பார்த்தேன்",("1","PL",""):"பார்த்தோம்",
        ("3","SG","M"):"பார்த்தான்",("3","SG","F"):"பார்த்தாள்",
        ("3","SG","H"):"பார்த்தார்",("3","PL",""):"பார்த்தார்கள்",
    },
    ("ஓடு","past"): {
        ("1","SG",""):"ஓடினேன்",("1","PL",""):"ஓடினோம்",
        ("3","SG","M"):"ஓடினான்",("3","SG","F"):"ஓடினாள்",
        ("3","SG","H"):"ஓடினார்",("3","PL",""):"ஓடினார்கள்",
    },
    ("படி","future"): {
        ("1","SG",""):"படிப்பேன்",("1","PL",""):"படிப்போம்",
        ("3","SG","M"):"படிப்பான்",("3","SG","F"):"படிப்பாள்",
        ("3","SG","H"):"படிப்பார்",("3","PL",""):"படிப்பார்கள்",
    },
    ("போ","future"): {
        ("1","SG",""):"போவேன்",("1","PL",""):"போவோம்",
        ("3","SG","M"):"போவான்",("3","SG","F"):"போவாள்",
        ("3","SG","H"):"போவார்",("3","PL",""):"போவார்கள்",
    },
    ("வா","future"): {
        ("1","SG",""):"வருவேன்",("1","PL",""):"வருவோம்",
        ("3","SG","M"):"வருவான்",("3","SG","F"):"வருவாள்",
        ("3","SG","H"):"வருவார்",("3","PL",""):"வருவார்கள்",
    },
    ("சாப்பிடு","future"): {
        ("1","SG",""):"சாப்பிடுவேன்",("1","PL",""):"சாப்பிடுவோம்",
        ("3","SG","M"):"சாப்பிடுவான்",("3","SG","F"):"சாப்பிடுவாள்",
        ("3","SG","H"):"சாப்பிடுவார்",("3","PL",""):"சாப்பிடுவார்கள்",
    },
}

PRESENT_ROOTS = ["படி","போ","வா","சாப்பிடு","எழுது","பார்","ஓடு","பேசு","விளையாடு","இரு"]
PAST_ROOTS    = ["படி","போ","வா","சாப்பிடு","எழுது","பார்","ஓடு"]
FUTURE_ROOTS  = ["படி","போ","வா","சாப்பிடு"]

PRONOUNS      = list(PRON_FEAT.keys())
SG_PRONOUNS   = ["நான்","நீ","அவன்","அவள்","அவர்","அது"]
PL_PRONOUNS   = ["நாங்கள்","நீங்கள்","அவர்கள்"]

LOC_NOUNS = ["பள்ளியில்","வீட்டில்","வகுப்பில்","சந்தையில்","மருத்துவமனையில்",
             "நூலகத்தில்","கோவிலில்","வயலில்","நகரத்தில்","காட்டில்"]
DAT_NOUNS = ["பள்ளிக்கு","வீட்டிற்கு","சந்தைக்கு","மருத்துவமனைக்கு",
             "அம்மாவிற்கு","அப்பாவிற்கு","நண்பனுக்கு","ஆசிரியருக்கு",
             "நூலகத்திற்கு","கோவிலுக்கு"]
ACC_NOUNS = ["புத்தகத்தை","கடிதத்தை","சாப்பாட்டை","தண்ணீரை",
             "பழத்தை","பையை","கணினியை","பாலை"]
ADVERBS   = ["தினமும்","இன்று","நேற்று","நாளை","விரைவாக",
             "மெதுவாக","நன்றாக","கவனமாக","எப்போதும்","அடிக்கடி"]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def tok2id(tok):
    return VOCAB.get(tok, (abs(hash(tok)) % 30000) + 1000)

def make(tokens, binary, error):
    assert len(tokens) == len(binary) == len(error)
    return {
        "tokens":        tokens,
        "input_ids":     [tok2id(t) for t in tokens],
        "attention_mask":[1]*len(tokens),
        "binary_labels": binary,
        "error_labels":  error,
    }

def get_verb(root, tense, pron):
    feat  = PRON_FEAT.get(pron, ("3","SG","M"))
    vdict = VERBS.get((root, tense), {})
    v = vdict.get(feat) or vdict.get((feat[0], feat[1], ""))
    if v is None:
        v = next(iter(vdict.values()), None)
    return v

def get_wrong_num_verb(root, tense, pron):
    """Verb with opposite number for SVA errors."""
    feat  = PRON_FEAT.get(pron, ("3","SG","M"))
    vdict = VERBS.get((root, tense), {})
    correct = get_verb(root, tense, pron)
    target_num = "PL" if feat[1] == "SG" else "SG"
    candidates = [f for (p,n,g),f in vdict.items() if n == target_num and f != correct]
    return random.choice(candidates) if candidates else None

def get_wrong_gender_verb(root, tense, pron):
    """Verb with opposite gender for ME errors."""
    feat  = PRON_FEAT.get(pron, ("3","SG","M"))
    vdict = VERBS.get((root, tense), {})
    correct = get_verb(root, tense, pron)
    opp = "F" if feat[2] == "M" else ("M" if feat[2] == "F" else None)
    if opp is None:
        return None
    candidates = [f for (p,n,g),f in vdict.items() if g == opp and f != correct]
    return random.choice(candidates) if candidates else None

# ─── Corpus builder ───────────────────────────────────────────────────────────
examples = []

# ── 1. CORRECT sentences (300) ────────────────────────────────────────────────
for _ in range(300):
    pron   = random.choice(PRONOUNS)
    tense  = random.choice(["present"]*4 + ["past"]*2 + ["future"])
    roots  = PRESENT_ROOTS if tense=="present" else (PAST_ROOTS if tense=="past" else FUTURE_ROOTS)
    root   = random.choice(roots)
    vf     = get_verb(root, tense, pron)
    if vf is None: continue

    pat = random.randint(0, 4)
    if pat == 0:
        toks = [pron, vf, "."]
    elif pat == 1:
        toks = [pron, random.choice(LOC_NOUNS), vf, "."]
    elif pat == 2:
        toks = [pron, random.choice(ADVERBS[:6]), vf, "."]
    elif pat == 3:
        toks = [pron, random.choice(ACC_NOUNS), vf, "."]
    else:
        toks = [pron, random.choice(ADVERBS[:5]), random.choice(LOC_NOUNS), vf, "."]

    n = len(toks)
    examples.append(make(toks, [0]*n, [0]*n))

# ── 2. SVA ERRORS — Subject-Verb Agreement (error_label=3, on verb) ───────────
for _ in range(110):
    pron   = random.choice(PRONOUNS)
    root   = random.choice(PRESENT_ROOTS[:6])
    wrong  = get_wrong_num_verb(root, "present", pron)
    if wrong is None: continue

    pat = random.randint(0, 2)
    if pat == 0:
        toks = [pron, wrong, "."];            vi = 1
    elif pat == 1:
        toks = [pron, random.choice(LOC_NOUNS), wrong, "."]; vi = 2
    else:
        toks = [pron, random.choice(ADVERBS[:5]), wrong, "."]; vi = 2

    bl = [0]*len(toks); el = [0]*len(toks)
    bl[vi] = 1; el[vi] = 3
    examples.append(make(toks, bl, el))

# ── 3. CME ERRORS — Case Marker Error (error_label=4, on noun) ────────────────
# Dative where locative expected (with படி/இரு verbs)
LOC_VERB_FORMS = {
    "படி": ["படிக்கிறேன்","படிக்கிறோம்","படிக்கிறான்","படிக்கிறாள்","படிக்கிறார்","படிக்கிறார்கள்",
            "படிக்கிறாய்","படிக்கிறீர்கள்"],
    "இரு": ["இருக்கிறேன்","இருக்கிறோம்","இருக்கிறான்","இருக்கிறாள்","இருக்கிறார்","இருக்கிறார்கள்"],
}
CME_PAIRS = [
    ("பள்ளிக்கு","பள்ளியில்"), ("வீட்டிற்கு","வீட்டில்"),
    ("நூலகத்திற்கு","நூலகத்தில்"), ("கோவிலுக்கு","கோவிலில்"),
    ("சந்தைக்கு","சந்தையில்"), ("மருத்துவமனைக்கு","மருத்துவமனையில்"),
]
for _ in range(90):
    pron       = random.choice(PRONOUNS)
    vroot      = random.choice(["படி","இரு"])
    vf         = random.choice(LOC_VERB_FORMS[vroot])
    wrong_n, _ = random.choice(CME_PAIRS)

    pat = random.randint(0, 1)
    if pat == 0:
        toks = [pron, wrong_n, vf, "."]; ni = 1
    else:
        toks = [pron, random.choice(ADVERBS[:5]), wrong_n, vf, "."]; ni = 2

    bl = [0]*len(toks); el = [0]*len(toks)
    bl[ni] = 1; el[ni] = 4
    examples.append(make(toks, bl, el))

# ── 4. TE ERRORS — Tense Error (error_label=2, on verb) ──────────────────────
for _ in range(90):
    pron    = random.choice(PRONOUNS)
    kind    = random.randint(0, 1)
    if kind == 0:   # future adv + past verb
        adv   = "நாளை"
        root  = random.choice(PAST_ROOTS[:5])
        vf    = get_verb(root, "past", pron)
    else:           # past adv + future verb
        adv   = "நேற்று"
        root  = random.choice(FUTURE_ROOTS)
        vf    = get_verb(root, "future", pron)
    if vf is None: continue

    pat = random.randint(0, 1)
    if pat == 0:
        toks = [adv, pron, vf, "."]; vi = 2
    else:
        toks = [adv, pron, random.choice(LOC_NOUNS), vf, "."]; vi = 3

    bl = [0]*len(toks); el = [0]*len(toks)
    bl[vi] = 1; el[vi] = 2
    examples.append(make(toks, bl, el))

# ── 5. ME ERRORS — Morphological suffix/gender mismatch (error_label=1) ───────
for _ in range(90):
    pron   = random.choice(["அவன்","அவள்","அவர்"])
    root   = random.choice(PRESENT_ROOTS[:6])
    wrong  = get_wrong_gender_verb(root, "present", pron)
    if wrong is None: continue

    pat = random.randint(0, 2)
    if pat == 0:
        toks = [pron, wrong, "."]; vi = 1
    elif pat == 1:
        toks = [pron, random.choice(LOC_NOUNS), wrong, "."]; vi = 2
    else:
        toks = [pron, random.choice(ADVERBS[:5]), wrong, "."]; vi = 2

    bl = [0]*len(toks); el = [0]*len(toks)
    bl[vi] = 1; el[vi] = 1
    examples.append(make(toks, bl, el))

# ── 6. WOE ERRORS — Word Order Error (error_label=6, verb first) ──────────────
for _ in range(60):
    pron   = random.choice(PRONOUNS)
    root   = random.choice(PRESENT_ROOTS[:6])
    vf     = get_verb(root, "present", pron)
    if vf is None: continue

    loc  = random.choice(LOC_NOUNS)
    toks = [vf, pron, loc, "."]   # verb before subject — SOV violation
    bl   = [1, 0, 0, 0]
    el   = [6, 0, 0, 0]
    examples.append(make(toks, bl, el))

# ── 7. NAE ERRORS — Number Agreement Error (error_label=7) ───────────────────
NAE_PAIRS = [("மாணவன்","மாணவர்கள்"),("ஆசிரியர்","ஆசிரியர்கள்"),
             ("நண்பன்","நண்பர்கள்"),("குழந்தை","குழந்தைகள்")]
for _ in range(60):
    pl_pron   = random.choice(PL_PRONOUNS)
    sg_noun,_ = random.choice(NAE_PAIRS)
    root      = random.choice(PRESENT_ROOTS[:6])
    vf        = get_verb(root, "present", pl_pron)
    if vf is None: continue

    toks = [pl_pron, sg_noun, vf, "."]
    bl   = [0, 1, 0, 0]
    el   = [0, 7, 0, 0]
    examples.append(make(toks, bl, el))

# ── 8. SV ERRORS — Sandhi Violation (error_label=5) ──────────────────────────
SV_TOKS = [
    "பள்ளி இல்","வீடு இல்","சந்தை இல்","நூலகம் இல்",
    "படிக்கிறான் கள்","வருகிறான் கள்","போகிறான் கள்",
    "பள்ளி க்கு","வீடு க்கு",
]
for _ in range(50):
    pron    = random.choice(PRONOUNS)
    sv_tok  = random.choice(SV_TOKS)
    toks    = [pron, sv_tok, "."]
    bl      = [0, 1, 0]
    el      = [0, 5, 0]
    examples.append(make(toks, bl, el))

# ── 9. SPE ERRORS — Spelling Error (error_label=8) ───────────────────────────
SPE_TOKS = [
    "பல்லி","மாணவன","அவர்கல்","படிக்றான்","வருகிரான்",
    "ஆசிரியர","பார்கிறான்","போகிரான்","சாப்பிடுகிரான்","நாங்கல்",
]
for _ in range(40):
    pron   = random.choice(PRONOUNS)
    spe_t  = random.choice(SPE_TOKS)
    toks   = [pron, spe_t, "."]
    bl     = [0, 1, 0]
    el     = [0, 8, 0]
    examples.append(make(toks, bl, el))

# ─── Shuffle and split 83/17 ─────────────────────────────────────────────────
random.shuffle(examples)
split          = int(0.83 * len(examples))
train_examples = examples[:split]
test_examples  = examples[split:]

# ─── Stats ───────────────────────────────────────────────────────────────────
err_names = {1:"ME",2:"TE",3:"SVA",4:"CME",5:"SV",6:"WOE",7:"NAE",8:"SPE"}
def stats(exs, label):
    dist = Counter(e for ex in exs for e in ex["error_labels"] if e > 0)
    tok_total = sum(len(ex["tokens"]) for ex in exs)
    err_total = sum(sum(b) for ex in exs for b in [ex["binary_labels"]])
    print(f"\n{label} ({len(exs)} sentences, {tok_total} tokens, {err_total} error tokens)")
    for k in sorted(dist): print(f"  {err_names.get(k,k):5s}: {dist[k]}")

stats(train_examples, "TRAIN")
stats(test_examples,  "TEST")

# ─── Save ─────────────────────────────────────────────────────────────────────
with open("data/train.jsonl","w",encoding="utf-8") as f:
    for ex in train_examples:
        f.write(json.dumps(ex, ensure_ascii=False)+"\n")

with open("data/test.jsonl","w",encoding="utf-8") as f:
    for ex in test_examples:
        f.write(json.dumps(ex, ensure_ascii=False)+"\n")

print(f"\n[✓] data/train.jsonl — {len(train_examples)} examples")
print(f"[✓] data/test.jsonl  — {len(test_examples)} examples")
