"""Minimal Gemini REST client (stdlib only). One fixed model + one embedding model, temperature 0."""
from __future__ import annotations

import json
import time

import common as c

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


_last_call = [0.0]


def _throttle() -> None:
    """Free tier has a low requests-per-minute limit; space calls out (config: gemini_min_interval_sec)."""
    gap = float(c.CONFIG.get("gemini_min_interval_sec", 0))
    wait = _last_call[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _headers(throttle: bool = True) -> dict:
    if throttle:  # embeddings have their own, higher quota
        _throttle()
    return {"x-goog-api-key": c.require_key("GEMINI_API_KEY"), "Content-Type": "application/json"}


_thinking_rejected = [False]


def _thinking_config() -> dict | None:
    """Gemini 2.5 takes thinkingBudget; Gemini 3+ takes thinkingLevel. Set one of them in config.json."""
    if _thinking_rejected[0] or c.CONFIG.get("thinking_rejected_by_model"):
        return None  # model refused the setting during the run, so it ran at its default; don't resend it
    if c.CONFIG.get("thinking_level"):
        return {"thinkingLevel": c.CONFIG["thinking_level"]}
    if "thinking_budget" in c.CONFIG:
        return {"thinkingBudget": int(c.CONFIG["thinking_budget"])}
    return None


def settings_tag() -> str:
    """Generation settings that change outputs; part of every translate/judge cache key."""
    if c.CONFIG.get("thinking_level"):
        return f"tl{c.CONFIG['thinking_level']}"
    return f"tb{c.CONFIG.get('thinking_budget', 'default')}"


def generate(prompt: str, json_mode: bool = False) -> str:
    cfg = {"temperature": c.CONFIG["temperature"], "maxOutputTokens": int(c.CONFIG.get("max_output_tokens", 8192))}
    tc = _thinking_config()
    if tc:
        cfg["thinkingConfig"] = tc
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
    url = f'{BASE}/{c.CONFIG["gemini_model"]}:generateContent'
    try:
        resp = c.http_json(url, headers=_headers(), body=json.dumps(body).encode(), timeout=420)
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("HTTP 404") or "no longer available" in msg:
            raise SystemExit(f"Model {c.CONFIG['gemini_model']} cannot be used with this key:\n{msg[:400]}\n"
                             "Nothing was spent. Change gemini_model in config.json, then run "
                             "`py src/check_setup.py --live` before the real run.")
        if msg.startswith("HTTP 400") and "thinking" in msg.lower() and tc:
            # this model rejects that thinking setting: retry once without it (logged, so it is never silent)
            c.log(f"WARNING: {c.CONFIG['gemini_model']} rejected thinkingConfig {tc}; using the model default.")
            _thinking_rejected[0] = True
            return generate(prompt, json_mode)
        raise
    cand = (resp.get("candidates") or [{}])[0]
    if cand.get("finishReason") == "MAX_TOKENS":
        return ""  # truncated reply: caller treats it as malformed and splits the batch
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response: {json.dumps(resp)[:400]}") from None


def embed(text: str) -> list[float]:
    model = c.CONFIG["embedding_model"]
    body = {"model": f"models/{model}", "content": {"parts": [{"text": text}]}, "taskType": "SEMANTIC_SIMILARITY"}
    resp = c.http_json(f"{BASE}/{model}:embedContent", headers=_headers(throttle=False), body=json.dumps(body).encode())
    return resp["embedding"]["values"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Up to 100 texts in one request (batchEmbedContents) to stay inside free-tier request limits."""
    model = c.CONFIG["embedding_model"]
    reqs = [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}, "taskType": "SEMANTIC_SIMILARITY"}
            for t in texts]
    resp = c.http_json(f"{BASE}/{model}:batchEmbedContents", headers=_headers(throttle=False),
                       body=json.dumps({"requests": reqs}).encode())
    out = [e["values"] for e in resp["embeddings"]]
    if len(out) != len(texts):
        raise RuntimeError(f"batchEmbedContents returned {len(out)} vectors for {len(texts)} texts")
    return out


def generate_json_list(prompt: str, n: int) -> list[dict] | None:
    """Batched call that must return a JSON array of n objects with integer `id` 1..n; None if malformed."""
    raw = generate(prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        return None
    by_id = {}
    for item in data:
        try:
            by_id[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    if sorted(by_id) != list(range(1, n + 1)):
        return None
    return [by_id[i] for i in range(1, n + 1)]


def run_batched(items: list, size: int, call, on_done=None) -> None:
    """Send items in chunks of `size`. call(chunk) returns a list aligned with chunk, or None if the reply was
    malformed. A malformed chunk is split in half and retried, so one bad reply costs ~2 extra requests,
    not one per item."""
    def go(chunk):
        res = call(chunk)
        if res is not None:
            if on_done:
                on_done(len(chunk))
            return
        if len(chunk) == 1:
            raise RuntimeError("Gemini returned a malformed reply for a single item; inspect the prompt.")
        mid = len(chunk) // 2
        go(chunk[:mid]); go(chunk[mid:])

    for i in range(0, len(items), size):
        go(items[i:i + size])


def list_models() -> list[str]:
    resp = c.http_json(f"{BASE}?pageSize=200", headers=_headers(), method="GET")
    return [m["name"] for m in resp.get("models", [])]
