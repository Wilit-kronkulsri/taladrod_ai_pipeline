"""Vehicle body-type categorization.

TaladRod does not publish a body-type field (see docs/recon.md), so the order is:
  1. ``body_type_raw`` if a future page/source provides one  -> source "site"
  2. a hand-built dictionary of models sold in Thailand (longest name wins, whole-word match)
     plus generic body words the seller writes ("กระบะ", "CAB", "SUV", ...) -> source "dictionary"
  3. otherwise "อื่นๆ"                                         -> source "fallback"
The categories follow how Thai buyers shop: เก๋ง, กระบะ, SUV/PPV, รถตู้/MPV, อื่นๆ.
"""
from __future__ import annotations

import re

from .models import AdRecord

SEDAN, PICKUP, SUV, VAN, OTHER = "เก๋ง", "กระบะ", "SUV/PPV", "รถตู้/MPV", "อื่นๆ"

MODEL_CATEGORY: dict[str, str] = {}


def _add(category: str, names: str) -> None:
    for n in names.split("|"):
        MODEL_CATEGORY[n.strip().upper()] = category


_add(PICKUP, "HILUX|REVO|VIGO|TIGER|D-MAX|DMAX|D MAX|RANGER|RAPTOR|NAVARA|TRITON|BT-50|BT50|COLORADO|"
             "TUNLAND|CANNON|POER|RIDER|NP300|FRONTIER|STRADA|L200|SPARK|DRAGON EYE|EXTENDER|T60|"
             "BIG M|CARRY|TATA XENON")
_add(VAN, "COMMUTER|HIACE|VENTURY|MAJESTY|GRANVIA|ALPHARD|VELLFIRE|STARIA|H-1|H1|GRAND STAREX|"
          "CARNIVAL|SERENA|URVAN|FREED|ODYSSEY|STEPWGN|MOBILIO|SIENTA|INNOVA|AVANZA|VELOZ|WISH|"
          "ESTIMA|PREVIA|XPANDER|ERTIGA|APV|SPACE WAGON|LM|V-CLASS|VITO|CARAVELLE|MULTIVAN|"
          "SPACETOURER|TRAVELLER|M8|G10|ZHIDOU|SPACEGEAR")
_add(SUV, "FORTUNER|MU-X|MUX|MU-7|MU7|PAJERO|PAJERO SPORT|EVEREST|CR-V|CRV|HR-V|HRV|BR-V|BRV|WR-V|"
          "CX-3|CX-30|CX-5|CX-8|CX-60|COROLLA CROSS|YARIS CROSS|C-HR|CHR|RAV4|LAND CRUISER|PRADO|"
          "HARRIER|RUSH|RAIZE|X-TRAIL|XTRAIL|KICKS|TERRA|JUKE|OUTLANDER|ASX|XFORCE|TUCSON|SANTA FE|"
          "CRETA|SPORTAGE|SORENTO|SELTOS|STONIC|CAPTIVA|TRAILBLAZER|ECOSPORT|TERRITORY|FORESTER|XV|"
          "CROSSTREK|HAVAL|H6|JOLION|TANK|TANK 300|TANK 500|ORA 07|ZS|HS|VS|EXTENDER SUV|ATTO 3|SEALION|"
          "5 EV|J7|OMODA|TIGGO|AION Y|AION V|DEEPAL S07|X1|X2|X3|X4|X5|X6|X7|IX|IX1|IX3|"
          "GLA|GLB|GLC|GLE|GLS|G-CLASS|EQA|EQB|EQC|Q2|Q3|Q5|Q7|Q8|NX|RX|UX|LX|XC40|XC60|XC90|"
          "MACAN|CAYENNE|COUNTRYMAN|RANGE ROVER|DISCOVERY|DEFENDER|EVOQUE|VELAR|MODEL Y|MODEL X|"
          "CAPTUR|KOLEOS|ECLIPSE CROSS|JIMNY|VITARA|S-CROSS|XL7|WRANGLER|COMPASS|CHEROKEE")
_add(SEDAN, "CIVIC|CITY|ACCORD|JAZZ|BRIO|AMAZE|YARIS|YARIS ATIV|ATIV|VIOS|ALTIS|COROLLA ALTIS|CAMRY|"
            "PRIUS|86|GR86|SUPRA|ALMERA|SYLPHY|TEANA|PULSAR|MARCH|NOTE|LEAF|MAZDA2|MAZDA 2|MAZDA3|"
            "MAZDA 3|MAZDA6|MX-5|MIRAGE|ATTRAGE|LANCER|LANCER EX|SWIFT|CIAZ|CELERIO|CRUZE|SONIC|AVEO|"
            "OPTRA|FIESTA|FOCUS|MUSTANG|ELANTRA|IONIQ|ELANTRA|GOOD CAT|ORA GOOD CAT|DOLPHIN|SEAL|"
            "MG3|MG 3|MG5|MG 5|MG4|MG 4|CYBERSTER|NEON|SERIES 1|SERIES 2|SERIES 3|SERIES 4|SERIES 5|"
            "SERIES 6|SERIES 7|SERIES 8|Z4|I4|I5|I7|A-CLASS|C-CLASS|E-CLASS|S-CLASS|CLA|CLS|CLE|"
            "AMG GT|SLK|SLC|EQE|EQS|A3|A4|A5|A6|A8|TT|IS|ES|GS|LS|LC|RC|CT|S60|S90|V40|V60|"
            "PANAMERA|TAYCAN|911|BOXSTER|CAYMAN|COOPER|MINI|CLUBMAN|MODEL 3|MODEL S|BEETLE|GOLF|"
            "PASSAT|POLO|IMPREZA|LEVORG|WRX|BRZ|CORVETTE|K5|FORTE|RIO")

# Generic words sellers use in titles/headlines (Thai and English).
BODY_WORDS: dict[str, str] = {
    "กระบะ": PICKUP, "แคป": PICKUP, "แค็บ": PICKUP, "CAB": PICKUP, "PICKUP": PICKUP, "PICK UP": PICKUP,
    "ตอนเดียว": PICKUP, "รถตู้": VAN, "MPV": VAN, "VAN": VAN, "SUV": SUV, "PPV": SUV,
    "เก๋ง": SEDAN, "SEDAN": SEDAN, "HATCHBACK": SEDAN, "COUPE": SEDAN,
}

# What a site-provided body label would map to.
SITE_BODY_MAP: dict[str, str] = {
    "รถเก๋ง": SEDAN, "เก๋ง": SEDAN, "sedan": SEDAN, "hatchback": SEDAN, "coupe": SEDAN,
    "รถกระบะ": PICKUP, "กระบะ": PICKUP, "pickup": PICKUP,
    "suv": SUV, "ppv": SUV, "รถอเนกประสงค์": SUV,
    "รถตู้": VAN, "ตู้": VAN, "van": VAN, "mpv": VAN,
}


def _pattern(term: str) -> re.Pattern:
    # whole-word match; "-", "." and spaces inside the term are literal
    return re.compile(r"(?<![A-Z0-9ก-๙])" + re.escape(term) + r"(?![A-Z0-9ก-๙])")


_MODEL_PATTERNS = sorted(((len(k), k, _pattern(k), v) for k, v in MODEL_CATEGORY.items()), reverse=True)
_BODY_PATTERNS = sorted(((len(k), k, _pattern(k.upper()), v) for k, v in BODY_WORDS.items()), reverse=True)


def _from_site(body: str | None) -> str | None:
    if not body:
        return None
    key = body.strip().lower()
    for word, cat in SITE_BODY_MAP.items():
        if word.lower() in key:
            return cat
    return None


def categorize(ad: AdRecord) -> tuple[str, str]:
    """Return ``(category, category_source)`` for one ad."""
    site = _from_site(ad.body_type_raw)
    if site:
        return site, "site"

    name = f" {ad.full_title} {ad.model or ''} ".upper()
    model_only = f" {ad.model or ''} ".upper()
    for length, _, pat, cat in _MODEL_PATTERNS:
        # 1-2 character codes (GS, ZS, LM, X1...) collide with trim names, so only trust
        # them when they are the model field itself
        haystack = model_only if length <= 2 else name
        if pat.search(haystack):
            return cat, "dictionary"

    free_text = f" {ad.full_title} {ad.seller_headline or ''} ".upper()
    for _, _, pat, cat in _BODY_PATTERNS:
        if pat.search(free_text):
            return cat, "dictionary"

    return OTHER, "fallback"
