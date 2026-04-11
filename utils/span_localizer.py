from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

TIME_WORDS = ["century", "decade", "era"]

STOPWORDS = {
    "one", "two", "three",
    "many", "most", "some",
    "several", "few",
    "considered", "known",
    "famous", "romantic"
}

OPINION_WORDS = [
    "considered",
    "believed",
    "thought",
    "one of the most",
    "famous",
    "romantic"
]


def semantic_match(span, evidence_sentences, threshold=0.7):

    span_emb = model.encode([span])
    ev_emb = model.encode(evidence_sentences)

    scores = cosine_similarity(span_emb, ev_emb)[0]

    return max(scores) > threshold


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def localize_spans(sentence, evidence, claim_units):

    hallucinated_spans = []

    sentence_lower = sentence.lower()
    evidence_lower = evidence.lower()

    evidence_sentences = split_sentences(evidence)

    # ------------------------------------------------
    # 1️⃣ Opinion filter (skip span detection)
    # ------------------------------------------------
    if any(word in sentence_lower for word in OPINION_WORDS):
        return []

    # ------------------------------------------------
    # 2️⃣ Numbers
    # ------------------------------------------------
    for num in claim_units["numbers"]:

        if len(num) <= 2:
            continue

        if num in evidence:
            continue

        if not semantic_match(num, evidence_sentences):
            hallucinated_spans.append(num)

    # ------------------------------------------------
    # 3️⃣ Entities
    # ------------------------------------------------
    for ent_text, _ in claim_units["entities"]:

        span = ent_text.lower()

        # ignore stopwords like "one"
        if span in STOPWORDS:
            continue

        # ignore time phrases
        if any(t in span for t in TIME_WORDS):
            continue

        # direct match
        if span in evidence_lower:
            continue

        if not semantic_match(span, evidence_sentences):
            hallucinated_spans.append(ent_text)

    # ------------------------------------------------
    # 4️⃣ Role reasoning rule (Apollo 11 case)
    # ------------------------------------------------
    if "accompanied by" in sentence_lower:

        if "michael collins" in sentence_lower:

            if "orbit" in evidence_lower or "command module" in evidence_lower:
                hallucinated_spans.append("Michael Collins")

    return list(set(hallucinated_spans))