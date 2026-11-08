"""Deterministic critical-fact extractor and matcher (no AI).

Fact types, extracted from the English human reference (`raw_transcription` of the en_us row):
  number     digits (decimals, commas) and number words, normalised to a value ("forty" -> 40, "2 million" -> 2000000)
  date_time  month and weekday names, 4-digit years, clock times, ordinal dates
  negation   not, no, never, none, nobody, nothing, neither, nor, without, n't (one fact per clip)
  name_place capitalised tokens or spans that are not sentence-initial (sentence-initial kept only when the
             same word also appears capitalised elsewhere in the text), minus a stoplist
Matching in a translation: numbers by value; dates by normalised token; negation by any negation token;
names by normalised exact match, then a labelled fuzzy match (token ratio >= 0.85, fuzzy=true).

Four-state label per fact (T1 = translation of the clean transcript, T2 = translation of what Scribe heard):
  T1 pass, T2 pass -> survived
  T1 pass, T2 fail -> asr_associated_break
  T1 fail, T2 fail -> translation_baseline_break
  T1 fail, T2 pass -> compensation_or_discordant
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}
TENS = {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
# lakh and crore: South Asian English renders "two million" as "20 lakh"; same value, so it must match
SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100_000, "lakhs": 100_000, "million": 1_000_000,
          "crore": 10_000_000, "crores": 10_000_000, "billion": 1_000_000_000}
NUMWORDS = {**UNITS, **TENS}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without"}
# a translation can carry the same negation in a prefix: "not yet known" -> "still unknown"
NEG_PREFIXED = {"unknown", "unable", "unclear", "uncertain", "unaware", "unavailable", "unsafe", "unpaid", "unfinished",
                "untreated", "unresolved", "impossible", "incomplete", "inaccessible", "nonexistent", "nowhere"}
STOP = {"The", "A", "An", "In", "On", "At", "Of", "For", "To", "And", "But", "Or", "If", "When", "While", "As",
        "It", "Its", "This", "That", "These", "Those", "He", "She", "They", "We", "You", "I", "His", "Her",
        "Their", "Our", "Your", "My", "Some", "Many", "Most", "There", "Here", "So", "Yet", "Also", "However",
        "After", "Before", "During", "Although", "Because", "Since", "With", "From", "By", "All", "Any",
        "Each", "Every", "One", "Mr", "Mrs", "Ms", "Dr"}
FUZZY_MIN = 0.85
STATE_PRIORITY = ["asr_associated_break", "translation_baseline_break", "compensation_or_discordant", "survived"]


def _clean(t: str) -> str:
    return unicodedata.normalize("NFKC", t or "")


def _norm_name(t: str) -> str:
    t = _clean(t).lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens(text: str) -> list[str]:
    text = _clean(text).replace("’", "'")
    return re.findall(r"\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:'[A-Za-z]+)?|\S", text)


def _to_num(tok: str):
    t = tok.lower().rstrip("%")
    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", t):
        v = float(t.replace(",", ""))
        return int(v) if v.is_integer() else v
    return NUMWORDS.get(t)


def numbers_in(text: str, lenient: bool = False) -> list[tuple[str, float]]:
    """Number phrases with their combined value. lenient=True also returns each component (for matching)."""
    toks = _tokens(text)
    out, i = [], 0
    while i < len(toks):
        v = _to_num(toks[i])
        if v is None:
            i += 1
            continue
        start, total, cur = i, 0, v
        j = i + 1
        while j < len(toks):
            t = toks[j].lower()
            if t in SCALES:
                cur = (cur or 1) * SCALES[t]
                if SCALES[t] >= 1000:
                    total += cur; cur = 0
                j += 1
            elif t in UNITS and cur and cur % 100 in (20, 30, 40, 50, 60, 70, 80, 90) and UNITS[t] < 10:
                cur += UNITS[t]; j += 1          # "twenty five"
            elif t == "and" and j + 1 < len(toks) and _to_num(toks[j + 1]) is not None and cur and cur >= 100:
                j += 1                            # "two hundred and five"
            elif _to_num(toks[j]) is not None and cur and cur >= 100 and _to_num(toks[j]) < 100:
                cur += _to_num(toks[j]); j += 1
            else:
                break
        value = total + cur
        phrase = " ".join(toks[start:j])
        out.append((phrase, value))
        if lenient:
            out += [(toks[k], _to_num(toks[k])) for k in range(start, j) if _to_num(toks[k]) is not None]
        i = j
    return out


TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?=\W|$)|\b(\d{1,2}):(\d{2})\b", re.I)
ORD_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.I)
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def dates_in(text: str) -> list[tuple[str, str]]:
    t = _clean(text)
    out = []
    for m in TIME_RE.finditer(t):
        if m.group(4):
            h, mi = int(m.group(4)), int(m.group(5))
        else:
            h, mi = int(m.group(1)), int(m.group(2) or 0)
            pm = m.group(3).lower().startswith("p")
            if pm and h < 12:
                h += 12
            if not pm and h == 12:
                h = 0
        out.append((m.group(0).strip(), f"{h:02d}:{mi:02d}"))
    for m in ORD_RE.finditer(t):
        out.append((m.group(0), f"day{int(m.group(1))}"))
    for m in YEAR_RE.finditer(t):
        out.append((m.group(0), f"year{m.group(1)}"))
    words = re.findall(r"[A-Za-z]+", t)
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in WEEKDAYS:
            out.append((w, lw))
        elif lw in MONTHS:
            # "may" is usually a verb: count it only when capitalised mid-sentence
            if lw == "may" and not (w[0].isupper() and i > 0):
                continue
            out.append((w, lw))
    return out


def negations_in(text: str) -> list[str]:
    """Negation tokens. A double negative ("not uncommon", "not unusual") means the positive, so it is skipped:
    a translation saying "common" keeps the meaning and must not count as a lost negation."""
    t = _clean(text).lower().replace("’", "'")
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", t)
    found = [w for i, w in enumerate(words) if w in NEGATIONS
             and not (w == "not" and i + 1 < len(words) and re.match(r"un[a-z]{3,}", words[i + 1]))]
    found += [w for w in words if w.endswith("n't")]
    return found


def names_in(text: str) -> list[str]:
    t = _clean(text)
    t = re.sub(r"\b(?:[A-Z]\.){2,}", lambda m: m.group().replace(".", ""), t)   # U.S. -> US
    skip = set(MONTHS) | set(WEEKDAYS) | set(NUMWORDS) | set(SCALES)
    caps_mid = set()
    sentences = re.split(r'(?<=[a-z0-9"\')][.!?])\s+', t)
    for s in sentences:
        ws = re.findall(r"[A-Za-z][\w'-]*", s)
        for i, w in enumerate(ws):
            if i > 0 and w[0].isupper():
                caps_mid.add(w)
    out = []
    for s in sentences:
        ws = re.findall(r"[A-Za-z][\w'-]*", s)
        run: list[str] = []
        for i, w in enumerate(ws):
            if len(w) == 1 and w.isupper():
                continue                      # middle initials
            is_cap = w[0].isupper() and w not in STOP and w.lower() not in skip
            if i == 0 and is_cap and w not in caps_mid and not (len(w) >= 2 and w.isupper()):
                is_cap = False                # sentence-initial: only if capitalised elsewhere too
            if is_cap:
                run.append(re.sub(r"'s$", "", w))
            elif run:
                out.append(" ".join(run)); run = []
        if run:
            out.append(" ".join(run))
    return out


def extract(reference: str) -> list[dict]:
    """Critical facts in the English reference, deduplicated by (type, norm)."""
    facts, seen = [], set()
    date_tokens = dates_in(reference)
    year_vals = {d[1] for d in date_tokens if d[1].startswith("year")}
    time_digits = set()
    for text, norm in date_tokens:
        if norm[:2].isdigit() or norm.startswith("day"):
            time_digits.update(re.findall(r"\d+", text))
    for text, norm in date_tokens:
        key = ("date_time", norm)
        if key not in seen:
            seen.add(key); facts.append({"type": "date_time", "text": text, "norm": norm})
    low = _clean(reference).lower()
    for text, value in numbers_in(reference):
        if f"year{text}" in year_vals or text in time_digits:
            continue
        if text.lower() == "one" and re.search(r"\bone of\b", low):
            continue   # "one of the ..." is a phrase, not a quantity staff would record
        key = ("number", str(value))
        if key not in seen:
            seen.add(key); facts.append({"type": "number", "text": text, "norm": str(value)})
    neg = negations_in(reference)
    if neg:
        facts.append({"type": "negation", "text": ", ".join(dict.fromkeys(neg)), "norm": "neg"})
    for n in names_in(reference):
        key = ("name_place", _norm_name(n))
        if key not in seen and _norm_name(n):
            seen.add(key); facts.append({"type": "name_place", "text": n, "norm": _norm_name(n)})
    return facts


def _name_match(norm: str, cand: str) -> tuple[bool, bool]:
    """(matched, fuzzy). Exact = normalised phrase present on word boundaries; fuzzy = best window ratio >= 0.85."""
    c = _norm_name(cand)
    if not norm or not c:
        return False, False
    if re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", c):
        return True, False
    ft, ct = norm.split(), c.split()
    n = len(ft)
    best = 0.0
    for i in range(0, max(1, len(ct) - n + 1)):
        best = max(best, SequenceMatcher(None, norm, " ".join(ct[i:i + n])).ratio())
    return (best >= FUZZY_MIN), (best >= FUZZY_MIN)


def match(fact: dict, cand: str | None) -> tuple[bool | None, bool]:
    """(present, fuzzy). present=None when there is no candidate text (e.g. no hint run)."""
    if cand is None:
        return None, False
    if not cand.strip():
        return False, False
    t = fact["type"]
    if t == "number":
        vals = {str(v) for _, v in numbers_in(cand, lenient=True)}
        return fact["norm"] in vals, False
    if t == "date_time":
        return fact["norm"] in {d[1] for d in dates_in(cand)}, False
    if t == "negation":
        words = set(re.findall(r"[a-z]+", _clean(cand).lower()))
        return bool(negations_in(cand) or words & NEG_PREFIXED), False
    return _name_match(fact["norm"], cand)


def state(in_t1: bool | None, in_t2: bool | None) -> str | None:
    if in_t1 is None or in_t2 is None:
        return None
    if in_t1 and in_t2:
        return "survived"
    if in_t1 and not in_t2:
        return "asr_associated_break"
    if not in_t1 and not in_t2:
        return "translation_baseline_break"
    return "compensation_or_discordant"


def clip_state(states: list[str | None]) -> str | None:
    s = [x for x in states if x]
    if not s:
        return None
    return min(s, key=STATE_PRIORITY.index)


def check(reference: str, t1: str, t2_auto: str, t2_hint: str | None = None) -> list[dict]:
    out = []
    for f in extract(reference):
        a, fa = match(f, t1)
        b, fb = match(f, t2_auto)
        h, fh = match(f, t2_hint)
        out.append({**f, "in_t1": a, "in_t2_auto": b, "in_t2_hint": h, "fuzzy": bool(fa or fb or fh),
                    "state_auto": state(a, b), "state_hint": state(a, h)})
    return out


def change_kind(fact: dict, cand: str) -> str:
    """For a fact missing from `cand`: 'changed' if a different value of the same type is present, else 'missing'."""
    t = fact["type"]
    if t == "number" and numbers_in(cand):
        return "changed"
    if t == "date_time" and dates_in(cand):
        return "changed"
    return "missing"


def readback_facts(cand: str) -> list[dict]:
    """Critical facts an agent would read back from the English it received."""
    return extract(cand)
