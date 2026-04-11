def compute_hallucination_score(nli_probs, retrieval_confidence):

    Pc = nli_probs["contradiction"]
    Pn = nli_probs["neutral"]
    Pe = nli_probs["entailment"]

    score = (
        0.6 * Pc +
        0.3 * Pn * (1 - retrieval_confidence) +
        0.1 * (1 - Pe)
    )

    return float(min(max(score, 0), 1))
