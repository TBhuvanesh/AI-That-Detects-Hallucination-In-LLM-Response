import spacy
import re

class ClaimExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def extract_claim_units(self, sentence):
        doc = self.nlp(sentence)

        entities = [(ent.text, ent.label_) for ent in doc.ents]
        numbers = re.findall(r'\d+(?:\.\d+)?', sentence)

        return {
            "sentence": sentence,
            "entities": entities,
            "numbers": numbers
        }
