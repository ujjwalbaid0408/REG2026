"""Faithful local replica of the REG2026 final-report sub-metric.

Mirrors repo_template/submission_evaluation_code/evaluate_metrics.py
(REG25FinalReportEvaluator) exactly so we can measure report score offline on
the held-out split before spending a leaderboard submission.

    final_report_score = 0.15*(rouge + bleu) + 0.40*key + 0.30*emb

- bleu / rouge: self-contained pure-Python (identical to the official code:
  clipped 4-gram precision with NO brevity penalty; LCS-based ROUGE-L F1).
- key: Jaccard of de-duplicated, lower-cased scispaCy `en_core_sci_lg`
  entities (len >= 3).
- emb: PubMedBERT (NeuML/pubmedbert-base-embeddings) mean-pooled cosine,
  rescaled (s-0.5)/0.5 and clamped to [0, 1].

The embedding and keyword models load lazily so the cheap text metrics can be
used without them.
"""
from __future__ import annotations
from typing import Dict, List, Optional

# --- metric weights (must match evaluate_metrics.py) ---
REG25_TEXT_WEIGHT = 0.15
REG25_KEY_WEIGHT = 0.40
REG25_EMB_WEIGHT = 0.30
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
SPACY_MODEL = "en_core_sci_lg"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# --------------------------------------------------------------------------
# Text metrics — pure Python, identical to the official implementation.
# --------------------------------------------------------------------------
def bleu4(ref_text: str, hyp_text: str) -> float:
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    ref_4 = [" ".join(ref_words[i:i + 4]) for i in range(len(ref_words) - 3)]
    hyp_4 = [" ".join(hyp_words[i:i + 4]) for i in range(len(hyp_words) - 3)]
    count = total = 0
    for g in hyp_4:
        count += min(hyp_4.count(g), ref_4.count(g))
        total += 1
    if total == 0:
        return 0.0
    return clamp01(count / total)


def rouge_l(ref_text: str, hyp_text: str) -> float:
    def lcs(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                L[i][j] = L[i - 1][j - 1] + 1 if X[i - 1] == Y[j - 1] else max(L[i - 1][j], L[i][j - 1])
        return L[m][n]
    ref = ref_text.lower().split()
    hyp = hyp_text.lower().split()
    l = lcs(ref, hyp)
    p = l / len(hyp) if hyp else 0.0
    r = l / len(ref) if ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return clamp01(f)


# --------------------------------------------------------------------------
# Keyword (scispaCy) and embedding (PubMedBERT) — lazy-loaded.
# --------------------------------------------------------------------------
class KeywordScorer:
    def __init__(self, model_name: str = SPACY_MODEL):
        import spacy
        self.nlp = spacy.load(model_name)

    def keywords(self, text: str, min_length: int = 3) -> List[str]:
        doc = self.nlp(text)
        return list({ent.text.lower() for ent in doc.ents if len(ent.text) >= min_length})

    def score(self, ref_text: str, hyp_text: str) -> float:
        a, b = set(self.keywords(ref_text)), set(self.keywords(hyp_text))
        union = len(a | b)
        return clamp01(len(a & b) / union) if union else 0.0


class EmbeddingScorer:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _embed(self, text: str):
        import torch
        with torch.no_grad():
            inp = self.tok(text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            out = self.model(**inp)
        return out.last_hidden_state.mean(dim=1).squeeze().numpy()

    def score(self, ref_text: str, hyp_text: str, scale: float = 0.5) -> float:
        from sklearn.metrics.pairwise import cosine_similarity
        s = float(cosine_similarity([self._embed(ref_text)], [self._embed(hyp_text)])[0][0])
        if scale != 0 and s > scale:
            s = (s - scale) / (1 - scale)
        return clamp01(s)


class ReportMetric:
    """Full report sub-metric. Pass with_models=False for text-only (bleu/rouge)."""

    def __init__(self, with_models: bool = True, with_key: Optional[bool] = None,
                 with_emb: Optional[bool] = None):
        wk = with_models if with_key is None else with_key
        we = with_models if with_emb is None else with_emb
        self.kw: Optional[KeywordScorer] = KeywordScorer() if wk else None
        self.emb: Optional[EmbeddingScorer] = EmbeddingScorer() if we else None

    def evaluate(self, ref_text: str, hyp_text: str) -> Dict[str, float]:
        bleu = bleu4(ref_text, hyp_text)
        rouge = rouge_l(ref_text, hyp_text)
        key = self.kw.score(ref_text, hyp_text) if self.kw else 0.0
        emb = self.emb.score(ref_text, hyp_text) if self.emb else 0.0
        total = REG25_TEXT_WEIGHT * (rouge + bleu) + REG25_KEY_WEIGHT * key + REG25_EMB_WEIGHT * emb
        return {
            "final_report_score": clamp01(total),
            "bleu": bleu, "rouge": rouge, "key": key, "emb": emb,
        }
