"""Fixed project settings, declared once. Values live in config.json so the pipeline and this module agree.

The semantic threshold (0.85) was declared in the build spec before the false-comfort results were computed;
0.80 and 0.90 are reported as sensitivity checks only.
"""
from common import CONFIG

SCRIBE_MODEL = CONFIG["scribe_model"]
GEMINI_MODEL = CONFIG["gemini_model"]
EMBEDDING_MODEL = CONFIG["embedding_model"]
TEMPERATURE = CONFIG["temperature"]
SEM_THRESHOLD = float(CONFIG["semantic_threshold"])
SEM_SENSITIVITY = [float(x) for x in CONFIG["semantic_threshold_sensitivity"]]
SEED = int(CONFIG["stats_seed"])
REPS = int(CONFIG["bootstrap_reps"])
JUDGE_PASS = {"same", "minor"}
NORMALISATION = "Unicode NFKC, lower-case, punctuation and symbols removed, whitespace collapsed; CER ignores spaces"
