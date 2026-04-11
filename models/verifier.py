# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from config.settings import NLI_MODEL

# class NLIVerifier:
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
#         self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
#         self.model.eval()

#     def verify(self, premise, hypothesis):

#         inputs = self.tokenizer(
#             premise,
#             hypothesis,
#             return_tensors="pt",
#             truncation=True
#         )

#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             probs = torch.softmax(outputs.logits, dim=1)[0]

#         return {
#             "contradiction": probs[0].item(),
#             "neutral": probs[1].item(),
#             "entailment": probs[2].item()
#         }





import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.settings import NLI_MODEL


class NLIVerifier:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()

    def verify(self, claim, evidence):

        # IMPORTANT: premise = evidence, hypothesis = claim
        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        contradiction = probs[0].item()
        neutral = probs[1].item()
        entailment = probs[2].item()

        # Decide label
        if entailment >= max(contradiction, neutral):
            label = "supported"

        elif contradiction >= max(entailment, neutral):
            label = "contradiction"

        else:
            label = "neutral"

        return {
            "label": label,
            "contradiction": contradiction,
            "neutral": neutral,
            "entailment": entailment
        }