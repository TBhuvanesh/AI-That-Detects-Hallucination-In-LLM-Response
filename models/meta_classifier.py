import os
import joblib
import numpy as np


class MetaClassifier:
    """
    Meta-classifier for hallucination detection.
    Takes feature dictionary as input and outputs hallucination probability.
    """

    def __init__(self, model_path="models/meta_model.pkl"):
        self.model_path = model_path
        self.model = None

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            print("⚠ Meta-classifier model not found. Running in fallback mode.")

        # IMPORTANT: Define fixed feature order
        self.feature_order = [
            "contradiction_prob",
            "neutral_prob",
            "entailment_prob",
            "relevance_score",
            "num_claim_units",
            "sentence_length"
        ]

    def prepare_features(self, feature_dict):
        """
        Convert feature dictionary into ordered numpy array.
        """
        feature_vector = []

        for feature_name in self.feature_order:
            value = feature_dict.get(feature_name, 0.0)
            feature_vector.append(float(value))

        return np.array(feature_vector).reshape(1, -1)

    def predict(self, feature_dict):
        """
        Returns hallucination probability between 0 and 1.
        """

        # If model not trained yet → fallback to manual scoring
        if self.model is None:
            return self.fallback_score(feature_dict)

        X = self.prepare_features(feature_dict)

        prob = self.model.predict_proba(X)[0][1]
        return float(prob)

    def fallback_score(self, feature_dict):
        """
        Fallback logic if trained model not available.
        Uses weighted NLI scoring (your current logic).
        """

        contradiction = feature_dict.get("contradiction_prob", 0)
        neutral = feature_dict.get("neutral_prob", 0)

        score = 0.7 * contradiction + 0.3 * neutral
        return float(score)