"""Three independent meaning measures, so our own AI judge is checked three ways.

1. Judge:      Gemini, temperature 0, fixed rubric same 100 / minor 75 / changed 35 / unrelated 0
2. Embedding:  cosine similarity of true English vs translation (Gemini embedding model)
3. Facts:      rule-based - digits, dates and capitalised names in the true English, present in translation?
"""
from __future__ import annotations

import json
import math
import re
import time

import common as c
import mock

# v2 (after the signal test): v1 graded loose paraphrase ("had been to" vs "had lived in") as "changed".
# The question is whether 311 staff would record a different fact or take a different action.
RUBRIC = """Labels:
- "same": staff would record the same facts and take the same action. Paraphrase, synonyms, different word
  order, loose wording or a slightly different nuance still count as "same".
- "minor": small loss that would not change what staff record or do (a dropped detail, adjective or tense).
- "changed": a fact staff would record is wrong or missing (a number, date, place, person, object or the
  action requested), so staff could act wrongly.
- "unrelated": the candidate does not convey the reference message at all, or is empty.
Judge meaning only; do not penalise grammar or style."""

JUDGE_PROMPT = """You are grading a translation for a city 311 phone line.
REFERENCE is what the resident actually meant. CANDIDATE is what city staff would read.
Choose exactly one label.
""" + RUBRIC + """
Respond with JSON only: {{"label": "same|minor|changed|unrelated", "reason": "<12 words max>"}}

REFERENCE: {ref}
CANDIDATE: {cand}"""


def judge(true_en: str, cand: str) -> dict:
    rubric = c.CONFIG["judge_rubric"]
    if not cand.strip():
        return {"label": "unrelated", "score": 0, "reason": "empty"}
    model = "mock" if c.is_mock() else c.CONFIG["gemini_model"]
    key = _judge_key(true_en, cand)
    hit = c.cache_get("judge", key)
    if hit is not None:
        return hit
    if c.is_mock():
        out = {"label": mock.judge(true_en, cand), "reason": "fake"}
    else:
        import gemini
        raw = gemini.generate(JUDGE_PROMPT.format(ref=true_en, cand=cand), json_mode=True)
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'"label"\s*:\s*"(\w+)"', raw)
            out = {"label": m.group(1) if m else "PARSE_ERROR", "reason": raw[:80]}
    label = str(out.get("label", "")).lower()
    if label not in rubric:
        raise RuntimeError(f"Judge returned an unknown label {label!r}; inspect cache and tighten the prompt.")
    res = {"label": label, "score": rubric[label], "reason": out.get("reason", ""), "model": model,
           "prompt_version": c.CONFIG["judge_prompt_version"]}
    c.cache_put("judge", key, res)
    return res


BATCH_JUDGE_PROMPT = """You are grading translations for a city 311 phone line.
In each numbered item, REFERENCE is what the resident actually meant and CANDIDATE is what city staff would read.
Grade every item independently, using exactly one label per item.
""" + RUBRIC + """
Respond with JSON only: a list of objects {{"id": <number>, "label": "same|minor|changed|unrelated", "reason": "<12 words max>"}}, one per item, same ids.

Items:
{items}"""


def _judge_key(true_en: str, cand: str) -> str:
    """Prompt version is part of the key, so a new rubric can never silently reuse old grades."""
    v = str(c.CONFIG["judge_prompt_version"])
    if c.is_mock():
        return c.key_hash("judge", "mock", f"v{v}", true_en, cand)
    import gemini
    return c.key_hash("judge", c.CONFIG["gemini_model"], f"v{v}", gemini.settings_tag(), true_en, cand)


def pending_judges(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen, todo = set(), []
    for ref, cand in pairs:
        k = _judge_key(ref, cand)
        if cand.strip() and k not in seen and c.cache_get("judge", k) is None:
            seen.add(k); todo.append((ref, cand))
    return todo


def prefetch_judges(pairs: list[tuple[str, str]]) -> None:
    """Grade (reference, candidate) pairs in batches; caches each under the key judge() reads."""
    if c.is_mock():
        return
    import gemini
    rubric = c.CONFIG["judge_rubric"]
    todo = pending_judges(pairs)
    if not todo:
        return
    size = int(c.CONFIG.get("judge_batch_size", 10))
    c.log(f"Judging {len(todo)} translations, up to {size} per request (~{-(-len(todo) // size)} Flash requests)")
    done = [0]

    def progress(n):
        done[0] += n
        c.log(f"  judged {done[0]}/{len(todo)}")

    def call(chunk):
        listing = "\n".join(json.dumps({"id": j + 1, "reference": r, "candidate": cd}, ensure_ascii=False)
                            for j, (r, cd) in enumerate(chunk))
        res = gemini.generate_json_list(BATCH_JUDGE_PROMPT.format(items=listing), len(chunk))
        if res is None or not all(str(x.get("label", "")).lower() in rubric for x in res):
            return None
        for (r, cd), x in zip(chunk, res):
            label = str(x["label"]).lower()
            c.cache_put("judge", _judge_key(r, cd), {"label": label, "score": rubric[label],
                                                     "reason": x.get("reason", ""), "model": c.CONFIG["gemini_model"],
                                                     "prompt_version": c.CONFIG["judge_prompt_version"],
                                                     "settings": gemini.settings_tag(),
                                                     "batched": True, "batch_size": len(chunk)})
        return res

    gemini.run_batched(todo, size, call, progress)


def _embed_key(text: str) -> str:
    model = "mock" if c.is_mock() else c.CONFIG["embedding_model"]
    return c.key_hash("embed", model, text)


def prefetch_embeddings(texts: list[str]) -> None:
    if c.is_mock():
        return
    import gemini
    todo = list(dict.fromkeys(t for t in texts if t.strip() and c.cache_get("embed", _embed_key(t)) is None))
    if not todo:
        return
    # Free tier counts every text in a batch as one request (100/min, 1,000/day), so pace the batches.
    size = int(c.CONFIG.get("embed_batch_size", 90))
    gap = float(c.CONFIG.get("embed_min_interval_sec", 0))
    c.log(f"Embedding {len(todo)} texts in batches of {size}" + (f", {gap:.0f}s apart" if gap else ""))
    for i in range(0, len(todo), size):
        if i and gap:
            time.sleep(gap)
        chunk = todo[i:i + size]
        c.log(f"  embedded {min(i + size, len(todo))}/{len(todo)}")
        for t, v in zip(chunk, gemini.embed_batch(chunk)):
            c.cache_put("embed", _embed_key(t), {"model": c.CONFIG["embedding_model"], "text": t, "v": v})


def embedding(text: str) -> list[float]:
    model = "mock" if c.is_mock() else c.CONFIG["embedding_model"]
    key = c.key_hash("embed", model, text)
    hit = c.cache_get("embed", key)
    if hit is not None:
        return hit["v"]
    if c.is_mock():
        v = mock.embed(text)
    else:
        import gemini
        v = gemini.embed(text)
    c.cache_put("embed", key, {"model": model, "text": text, "v": v})
    return v


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def similarity(true_en: str, cand: str) -> float:
    if not cand.strip():
        return 0.0
    return round(cosine(embedding(true_en), embedding(cand)), 4)


# --------------------------------------------------------------------------- facts

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000", "million": "1000000",
}
STOP_CAPS = {"I", "The", "A", "An", "It", "In", "On", "At", "This", "That", "These", "Those", "He", "She", "They",
             "We", "You", "But", "And", "Or", "If", "When", "While", "As", "Some", "Many", "Most", "There"}


def _tokens(text: str) -> list[str]:
    toks = []
    for w in re.findall(r"[\w'-]+", text.lower()):
        w = w.replace(",", "")
        toks.append(NUMBER_WORDS.get(w, w))
    return toks


NUMBER_FACT_WORDS = set(NUMBER_WORDS) - {"one", "zero"}  # "one of the..." is not a fact


def extract_facts(true_en: str) -> list[str]:
    """Numbers (digits or words), dates and capitalised names. Comparison normalises number words to digits."""
    facts: list[str] = []
    text = re.sub(r"\b(?:[A-Z]\.){2,}", lambda m: m.group().replace(".", ""), true_en)  # U.S. -> US
    for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        facts.append(n.replace(",", ""))
    for w in re.findall(r"[A-Za-z]+", text):
        if w.lower() in NUMBER_FACT_WORDS:
            facts.append(w.lower())
    for sent in re.split(r'(?<=[a-z0-9"\')][.!?])\s+', text):
        words = re.findall(r"[A-Za-z][\w'-]*", sent)
        run: list[str] = []
        for i, w in enumerate(words):
            if len(w) == 1 and w.isupper():  # middle initials: skip without breaking the name
                continue
            acronym = len(w) >= 2 and w.isupper()
            # sentence-initial words are capitalised anyway, so only acronyms count there
            is_name = (i > 0 or acronym) and w[0].isupper() and w not in STOP_CAPS and w.lower() not in NUMBER_WORDS
            if is_name:
                run.append(w)
            elif run:
                facts.append(" ".join(run)); run = []
        if run:
            facts.append(" ".join(run))
    seen, out = set(), []
    for f in facts:
        if f.lower() not in seen:
            seen.add(f.lower()); out.append(f)
    return out


def fact_present(fact: str, cand: str) -> bool:
    toks = set(_tokens(cand))
    ftoks = _tokens(fact)
    # names: allow simple English plural/possessive variation
    return all(t in toks or t.rstrip("s") in {x.rstrip("s") for x in toks} for t in ftoks)


def facts_check(true_en: str, from_true: str, from_heard: str) -> list[dict]:
    return [{"text": f, "survived_true": fact_present(f, from_true), "survived_heard": fact_present(f, from_heard)}
            for f in extract_facts(true_en)]
