# Pitch (90 s) and video (2:00)

Every number comes from `app/data/app_data.json`, which the pipeline generates.

## 90-second pitch

**[0:00 Hook]** *(Screen 1, Urdu card, press "Replay verified example")*
"When a resident calls 311 in Urdu, the message passes through an ear that turns speech into text, then a
translator, before staff read it in English. It can sound understood and still lose the one fact staff need."

**[0:15 How]**
"We used the same 40 sentences read by real people in 14 languages, so we always know what the caller meant. We
translate twice, from the clean transcript and from what the ear heard, and check every critical fact by fixed
rules: numbers, dates, names, and the word not."

**[0:35 Finding]** *(Screen 2, the red false-comfort cell)*
"20 of 322 recordings looked fine on meaning but lost a critical fact: 6 percent, interval 3 to 10. Most of those
were lost at translation. And the ear has its own silent error: it labelled 7 of 40 Urdu callers as Hindi, and
wrote 22 of 40 Cantonese callers in the wrong script. English: zero."

**[1:00 Fix]** *(Screen 3, Urdu)*
"Supplying the language fixed Urdu's label, 82 to 100 percent, but not Cantonese's script, and not the facts. A
confirmation readback would have shown the caller the wrong fact in 24 of 28 cases. That is a chance to catch it,
not a fix."

**[1:15 Close]** *(Screen 4)*
"For the City: check facts, not fluency; ask the language first; read critical facts back. One limit: read speech
is a best case. Next step: a pilot on consented calls."

## Two-minute video (B narrates, A drives)

| Time | On screen | Narration |
|---|---|---|
| 0:00 to 0:15 | Screen 1, Urdu replay: detected Hindi, wrong script | Hook |
| 0:15 to 0:40 | Trace cards and the Benchmark receipt | Answer key, two translations, fact rules |
| 0:40 to 1:10 | Screen 2: open the false-comfort cell, play one example | 20 of 322; mostly translation; ear mislabels |
| 1:10 to 1:35 | Screen 3: Urdu then Cantonese slopes; readback line; agent widget | Language supplied, readback |
| 1:35 to 1:50 | Screen 4 | Screening, not a forecast |
| 1:50 to 2:00 | Methods drawer | Limits and the next step |

Record at 1280×720, upload to YouTube as Public, and check that it plays in a private window.

## Judge Q&A drill

| Question | Answer |
|---|---|
| Isn't 6% small? | It is the share that looks fine while losing a fact, so nobody would notice it. Over about 1,000 calls a day, a small rate matters, but we do not forecast call failures: this is read speech. |
| Language effect or speaker effect? | Partly speaker: few speakers per language, and Urdu, Punjabi and Arabic are female only. Flagged in every receipt. |
| Gemini translates and judges. Biased? | That is why facts are checked by rules, not AI. In a human spot-check the judge agreed on 12 of 20, and every disagreement was the judge being more lenient than the human. One rater only, which we state. |
| Did you tune the rules to the data? | Once, after inspecting examples. Every change is unit-tested, listed in Methods, and each one reduced false alarms rather than adding failures. |
| Why is the threshold insensitive? | Embedding similarity is compressed (99% of clips at or above 0.90), so the judge decides. We say so on screen. |
| Cantonese in Simplified, really wrong? | The reference transcripts are Hong Kong Traditional, and Mandarin was written correctly in Simplified 40 of 40. The ear picks the script for the caller. |
| Bengali? | No audio in the pack: unmeasured, not zero. |
| Why not train a model? | The question is how production AI serves residents; testing it against an answer key answers that directly. |
