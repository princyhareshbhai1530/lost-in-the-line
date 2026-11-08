# Lost in the Line

A multilingual 311 request can sound understood while losing the one fact staff need: the address, the date, the
number, or the word "not". We measured it on real speech in 14 languages, against a human English answer key.

Sheridan Datathon 2026. Member A: data engine and evaluation. Member B: prototype, ElevenLabs, story.
ElevenLabs Scribe is the speech-to-text system we evaluate; Google Gemini translates and judges.

## Question

When a resident speaks a home language to a voice line, which critical facts survive to the English that staff
read, which part of the chain loses them, and does telling the ear the language up front help?

## Data (track pack only)

- **FLEURS:** the same 40 sentences read by real speakers in 14 languages (13 home languages plus English), 560
  clips. For every clip, the reference is the English row with the same `fleurs_id`: `raw_transcription` for
  fact extraction (it keeps capitals and digits), `transcription` for normalised text comparison.
- **2021 Census** (neighbourhoods), **Mississauga 311** daily calls 2017 to 2018 (cleaned), and the language
  crosswalk. Toronto 311 is used only as a list of request types. No other dataset is downloaded.

## Method

1. **Ear:** Scribe v2 on every clip with the language auto-detected (main condition). We score CER and WER with one
   normalisation (NFKC, lower-case, punctuation removed; CER ignores spaces). Language detection is its own metric
   (ISO 639-1 or 639-3 accepted), with a separate script check (for example Urdu written in Devanagari, or Cantonese
   in Simplified characters).
2. **Two translations per clip:** T1 from the clean source transcript, T2 from what Scribe heard (Gemini,
   temperature 0, fixed prompt).
3. **Critical facts, no AI** (`src/facts.py`): numbers (digits and number words, normalised to values, including
   lakh and crore), dates and times, negation, and names and places, extracted from the English reference.
4. **Four-state labels:** we compare each fact across the two translations and label it survived, ASR-associated
   break (kept from the clean transcript, lost from what Scribe heard), translation-baseline break (lost from both),
   or compensation. For meaning scores we report the incremental degradation after substituting the ASR
   transcript, not a causal share.
5. **False-Comfort Matrix** (`src/false_comfort.py`): a clip passes the meaning check when the Gemini judge says
   same or minor and embedding similarity is at least 0.85 (declared before results; 0.80 and 0.90 as
   sensitivity). Crossed with "every critical fact kept", this gives four cells: reassuring agreement, false
   comfort, harmless wording, broad failure.
6. **Power-Up Decision Lab** (`src/powerup.py`): the same audio re-run with the language supplied to Scribe, paired
   per clip. "Would confirmation have fired?": a readback fires when the English contains a critical fact; a
   potential silent failure is surfaced when the readback would show the caller a fact of the type that went wrong.
7. **Intervals** (`src/stats.py`): paired cluster bootstrap by sentence (`fleurs_id`), 2,000 reps, seed 42.
8. **Mississauga screening** (`src/screening.py`): benchmark-weighted language exposure (a screening index, not a
   forecast). For each neighbourhood, residents who speak a tested language most often at home, weighted by that
   language's measured critical-fact loss, per 1,000 residents, with coverage (the share of residents in tested
   languages).

## Findings (560 clips; 322 contain at least one critical fact)

| Finding | Number (95% CI) |
|---|---|
| Looked fine on meaning but lost a critical fact (false comfort) | **20 of 322 (6.2%, 2.8% to 9.6%)** |
| Of those 20, lost even from the clean transcript (translation baseline) vs lost only from what Scribe heard | 17 vs 3 |
| Urdu clips labelled as the wrong language (all as Hindi, written in Devanagari) | 7 of 40 (detection 82.5%, 70% to 93%) |
| Cantonese clips written in Mainland Simplified script instead of Hong Kong Traditional | 22 of 40 (script correct 45%, 30% to 60%) |
| Other wrong labels | Cantonese as Japanese 2/40; Punjabi as Bulgarian 1/40 and as Hindi 1/40 |
| English baseline | 0 wrong labels, 0 false comfort, CER 0.008 |
| CER when language and script were right vs wrong (home languages) | 0.026 vs 0.543 |
| Language supplied: Urdu right language and script | 82.5% to **100%** (paired change +7.5 to +30 pts); CER 0.215 to 0.036 |
| Language supplied: Cantonese right script | 40% to 52.5% (+2.5 to +22.5 pts); most clips still in Simplified |
| Language supplied: critical-fact survival | no measurable change (Urdu 97.6% to 95.2%, CI of change −7.9 to 0 pts) |
| Readback would show the caller a wrong fact of the right type | 24 of the 28 clips with a wrong fact (potential silent failures surfaced) |
| Highest screening exposure | DT Cooksville 17.4 per 1,000 (coverage 30.7%), Fairview 15.4 (31.1%), Churchill Meadows 15.1 (28.7%) |
| Mississauga 311 context | 1,006 calls per business day, 206 s handle time (501 business days) |

**What this means for the City.** The meaning check usually agrees with the facts (279 of 322 clips), but about 1 in
16 fact-bearing requests looks fine while losing a fact, mostly at translation. Separately, the ear silently
mislabels the language for Urdu and writes Cantonese in the wrong script; supplying the language fixes Urdu's label
and error rate, but not Cantonese's script, and not the facts. A confirmation readback would give the caller a
chance to catch most wrong facts. Nothing here is a forecast of failed calls.

## Evaluating the evaluator

- **Three signals:** Gemini judge vs embedding similarity, Spearman 0.45 (n = 1,080 translations); judge vs the
  rule-based fact check 0.10. Embedding similarity is compressed (99.4% of fact-bearing clips at or above 0.90), so
  the threshold never changed a cell and the judge effectively decides the meaning check.
- **Human spot-check:** 20 random end-to-end pairs (seed 42) in `results/handcheck.csv`, labelled by one team member
  (single rater, so human vs human agreement is not measured). **The AI judge agreed with the human on 12 of 20.**
  All 8 disagreements go one way: the human marked the English as changed where the judge said same or minor (the
  judge's only "changed" was also caught by the human). The judge is more lenient than a careful reader, which is
  why we do not trust a meaning score on its own and check facts by rule.
- **Changes we made, and why:** judge prompt v2 (v1 graded paraphrase as changed); the model moved from
  gemini-2.5-flash to gemini-3.8-flash when Google closed 2.5 to new accounts, so every final translation and grade
  comes from one model; the fact rules were refined once after inspecting examples (double negatives, prefixed
  negation such as "unknown", "one of" as a phrase, lakh and crore). Each refinement is unit-tested and reduced
  false alarms.

## Models (none trained by us)

| Role | Model |
|---|---|
| Ear | ElevenLabs Scribe v2 (`scribe_v2`) |
| Translator and judge | `gemini-3.8-flash`, temperature 0, model-default thinking (the requested minimal setting was rejected) |
| Embeddings | `gemini-embedding-001` |

Model names, prompts, thresholds and the bootstrap settings are written to `results/run_config.json`. Every API
response is cached on disk, keyed by clip, model and settings, so re-runs never re-pay.

## Run it

```bash
py -m unittest discover tests
```
```bash
py run_pipeline.py evidence
```
```bash
py server/app.py
```
Then open `http://127.0.0.1:8000`. Live recording needs `ELEVENLABS_API_KEY` and `GEMINI_API_KEY` set in that
terminal. Without them, or offline, every screen works from `app/data/app_data.json` and Screen 1 uses the verified
replay. To open with no server at all, double-click `app/index.html`.

To add languages to the language-supplied rerun (costs Scribe credits, about 11 per clip):
```bash
py run_pipeline.py hint hi_in,ta_in
```
then run `py run_pipeline.py evidence` again.

Earlier stages (`signal`, `full`, `fix`) produced the cached transcripts, translations and grades; see
`run_pipeline.py`. `py src/score.py --estimate` prints the Gemini requests a run needs before spending any.

## Repo

```
src/facts.py            deterministic critical-fact extractor and four-state labels
src/build_eval_table.py one row per clip with the English answer key (results/eval_table.*)
src/false_comfort.py    False-Comfort Matrix, language detection, per-language metrics
src/powerup.py          language-supplied rerun, paired change, readback analysis
src/screening.py        benchmark-weighted language exposure per neighbourhood
src/stats.py            paired cluster bootstrap by fleurs_id
src/handcheck.py        20-pair human spot-check
src/export_app_data.py  app/data/app_data.json and the audio the app plays
server/app.py           live Play the Line backend and static hosting (standard library only)
app/index.html          the prototype (offline-safe; no external scripts)
```

## Limits

- FLEURS is read speech, a best case; real calls are noisier.
- Few speakers per language; Urdu, Punjabi and Arabic are female only, Portuguese, Tagalog and Vietnamese male
  only, so a language effect may partly be a speaker effect.
- Dialects: FLEURS Portuguese is Brazilian, Spanish Latin American, Arabic Egyptian.
- The answer key drifts: home-language sentences are loose translations of the English, which inflates
  translation-baseline breaks.
- The same Gemini model translates and judges; we check it with rules, embeddings and the human spot-check.
  Temperature 0 is only mostly deterministic, which is why every response is cached.
- Census home-language columns are "spoken most often at home"; the legend does not say whether they are single
  responses, so counts may not be unique people. Counts are rounded to 5. Bengali has no audio: unmeasured, not zero.
- Intended use: deciding which part of a voice line to fix. Not for judging any speaker or community.

## Sources and licences

City of Mississauga Open Data (2021 Census; 311 Citizen Contact Centre Call Data 2017, 2018) · FLEURS (Google,
CC BY 4.0) · Speech Accent Archive (Weinberger, GMU, CC BY-NC-SA 4.0; not used in the demo) · MInDS-14 (PolyAI,
CC BY 4.0) · Speech-MASSIVE (FBK, CC BY-NC-SA 4.0) · Toronto 311 (Open Government Licence, Toronto) · ElevenLabs ·
Google Gemini.

## AI use

ElevenLabs Scribe and Gemini are the systems under test. The pipeline code was written with help from Claude
(Anthropic) and reviewed by the team. No model was trained.
