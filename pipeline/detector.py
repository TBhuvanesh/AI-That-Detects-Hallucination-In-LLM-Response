"""
Hallucination Detector
----------------------
Main pipeline that:
1. Splits response into sentences
2. Extracts claim units
3. Retrieves Wikipedia evidence
4. Performs NLI verification
5. Uses a meta-classifier to compute hallucination probability
6. Localizes hallucinated spans
7. Produces sentence-level and overall hallucination scores
"""

import numpy as np

from config.settings import TOP_K
from models.claim_extractor import ClaimExtractor
from models.meta_classifier import MetaClassifier
from models.verifier import NLIVerifier
from models.wiki_retriever import WikiRetriever
from utils.span_localizer import localize_spans

RELEVANCE_THRESHOLD = 0.35

class HallucinationDetector:
    """
    Core class that runs the hallucination detection pipeline.
    """

    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.retriever = WikiRetriever()
        self.verifier = NLIVerifier()
        self.meta_classifier = MetaClassifier()

    # ---------------------------------------------------
    # Main analysis function
    # ---------------------------------------------------

    def analyze(self, question: str, response: str):
        """
        Analyze a model response and detect hallucinations.

        Args:
            question (str): Original user question
            response (str): Model-generated response

        Returns:
            dict: sentence-level results + overall hallucination score
        """

        sentences = self.claim_extractor.split_sentences(response)

        results = []
        scores = []

        for sentence in sentences:

            if not sentence.strip():
                continue

            # --------------------------------------------
            # Step 1: Extract claim units
            # --------------------------------------------

            claim_units = self.claim_extractor.extract_claim_units(sentence)

            # --------------------------------------------
            # Step 2: Retrieve evidence
            # --------------------------------------------

            evidence, relevance_score = self.retriever.retrieve(sentence)

            if evidence is None or relevance_score < RELEVANCE_THRESHOLD:

                results.append({
                    "sentence": sentence,
                    "hallucinated_spans": [],
                    "score": 0.2,
                    "label": "not_verifiable",
                    "evidence": "No relevant evidence found on Wikipedia."
                })

                scores.append(0.2)
                continue

            # --------------------------------------------
            # Step 3: NLI verification
            # --------------------------------------------

            nli_probs = self.verifier.verify(sentence, evidence)

            #label = max(nli_probs, key=nli_probs.get)

            if nli_probs["entailment"] > 0.5:
                label = "entailment"

            elif nli_probs["contradiction"] > 0.65:
                label = "contradiction"

            else:
                label = "neutral"

            # --------------------------------------------
            # Step 4: Feature engineering
            # --------------------------------------------

            features = {
                "contradiction_prob": nli_probs["contradiction"],
                "neutral_prob": nli_probs["neutral"],
                "entailment_prob": nli_probs["entailment"],
                "relevance_score": relevance_score,
                "num_claim_units": len(claim_units),
                "sentence_length": len(sentence.split())
            }

            # --------------------------------------------
            # Step 5: Meta-classifier scoring
            # --------------------------------------------

            score = self.meta_classifier.predict(features)

            # --------------------------------------------
            # Step 6: Localize hallucinated spans
            # --------------------------------------------

            hallucinated_spans = localize_spans(
                sentence,
                evidence,
                claim_units
            )

            # --------------------------------------------
            # Step 7: Hallucination override logic
            # --------------------------------------------

            if hallucinated_spans:

                if label == "entailment":
                    label = "partial_hallucination"
                    score = max(score, 0.6)

                #elif label == "neutral":
                #    label = "contradiction"
                #    score = max(score, 0.9)
                elif label == "neutral":
                    label = "partial_hallucination"
                    score = max(score, 0.6)

            # --------------------------------------------
            # Step 8: Save results
            # --------------------------------------------

            results.append({
                "features": features,
                "sentence": sentence,
                "hallucinated_spans": hallucinated_spans,
                "score": float(score),
                "label": label,
                "evidence": evidence[:1000]
            })

            scores.append(score)

        # --------------------------------------------
        # Step 9: Compute overall hallucination score
        # --------------------------------------------

        overall_score = float(sum(scores) / len(scores)) if scores else 0

        return {
            "sentences": results,
            "overall_score": overall_score
        }