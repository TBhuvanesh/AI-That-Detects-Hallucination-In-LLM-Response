# import wikipedia
# from sentence_transformers import SentenceTransformer, util
# import torch

# class WikiRetriever:
#     def __init__(self):
#         self.embed_model = SentenceTransformer("BAAI/bge-large-en")

#     def search_wikipedia(self, query, top_k=3):
#         try:
#             pages = wikipedia.search(query, results=top_k)
#         except Exception:
#             return []

#         documents = []

#         for title in pages:
#             try:
#                 page = wikipedia.page(title)
#                 content = page.content[:5000]  # limit size
#                 paragraphs = content.split("\n")
#                 # Filter out empty paragraphs
#                 documents.extend([p.strip() for p in paragraphs if p.strip()])
#             except Exception:
#                 continue

#         return documents

#     def retrieve(self, query, top_k=3):
#         documents = self.search_wikipedia(query, top_k)

#         if not documents:
#             return None, 0.0

#         try:
#             query_embedding = self.embed_model.encode(query, convert_to_tensor=True)
#             doc_embeddings = self.embed_model.encode(documents, convert_to_tensor=True)

#             scores = util.cos_sim(query_embedding, doc_embeddings)[0]

#             top_idx = torch.argmax(scores).item()
#             best_score = scores[top_idx].item()
#             best_doc = documents[top_idx]

#             return best_doc, best_score
#         except Exception as e:
#             print(f"Error during retrieval: {e}")
#             return None, 0.0









import wikipedia
import re
from sentence_transformers import SentenceTransformer, util
import torch


def extract_numbers(text):
    return re.findall(r'\d+', text)


def filter_relevant_sentences(query, sentences):
    query_words = set(re.findall(r'\w+', query.lower()))

    relevant = []

    for s in sentences:
        s_words = set(re.findall(r'\w+', s.lower()))

        overlap = query_words.intersection(s_words)

        if len(overlap) >= 2:
            relevant.append(s)

    return relevant if relevant else sentences

def clean_text(text):
    text = re.sub(r"==.*?==", "", text)
    text = text.replace("\n", " ")
    return text


def extract_keywords(query):
    stopwords = {"is", "the", "a", "an", "of", "in", "at", "when", "was", "on"}
    words = re.findall(r'\w+', query.lower())
    keywords = [w for w in words if w not in stopwords]
    return " ".join(keywords[:4])


class WikiRetriever:

    def __init__(self):
        self.embed_model = SentenceTransformer("all-mpnet-base-v2")

    def search_wikipedia(self, query, top_k=5):
        try:
            keywords = extract_keywords(query)
            pages = wikipedia.search(keywords, results=top_k)
        except Exception:
            return []

        documents = []

        for title in pages:
            try:
                page = wikipedia.page(title, auto_suggest=False)

                content = clean_text(page.content[:5000])

                # Better sentence splitting
                sentences = re.split(r'(?<=[.!?]) +', content)

                sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

                # NEW STEP
                filtered_sentences = filter_relevant_sentences(query, sentences)

                documents.extend(filtered_sentences)

            except Exception:
                continue

        return documents

    # def retrieve(self, query, top_k=5):

    #     documents = self.search_wikipedia(query, top_k)

    #     if not documents:
    #         return None, 0.0

    #     # extract important words from claim
    #     query_words = set(re.findall(r'\w+', query.lower()))

    #     # keep sentences sharing keywords
    #     filtered_docs = []

    #     for doc in documents:
    #         doc_words = set(re.findall(r'\w+', doc.lower()))
    #         overlap = query_words.intersection(doc_words)

    #         if len(overlap) >= 2:
    #             filtered_docs.append(doc)

    #     # fallback if filtering removes everything
    #     if not filtered_docs:
    #         filtered_docs = documents

    #     try:
    #         query_embedding = self.embed_model.encode(query, convert_to_tensor=True)
    #         doc_embeddings = self.embed_model.encode(filtered_docs, convert_to_tensor=True)

    #         scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    #         top_idx = torch.argmax(scores).item()
    #         best_score = scores[top_idx].item()
    #         best_doc = filtered_docs[top_idx]

    #         return best_doc, best_score

    #     except Exception as e:
    #         print(f"Error during retrieval: {e}")
    #         return None, 0.0




    def retrieve(self, query, top_k=5):

        documents = self.search_wikipedia(query, top_k)

        if not documents:
            return None, 0.0

        # extract important words from claim
        query_words = set(re.findall(r'\w+', query.lower()))

        # keep sentences sharing keywords
        filtered_docs = []

        for doc in documents:
            doc_words = set(re.findall(r'\w+', doc.lower()))
            overlap = query_words.intersection(doc_words)

            if len(overlap) >= 2:
                filtered_docs.append(doc)

        # fallback if filtering removes everything
        if not filtered_docs:
            filtered_docs = documents

        try:
            query_embedding = self.embed_model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.embed_model.encode(filtered_docs, convert_to_tensor=True)

            scores = util.cos_sim(query_embedding, doc_embeddings)[0]

            # -------------------------------
            # 🔥 FIX 2 STARTS HERE
            # -------------------------------

            top_indices = torch.topk(scores, k=min(5, len(scores))).indices

            best_doc = None
            best_score = 0

            query_numbers = extract_numbers(query)

            for idx in top_indices:
                doc = filtered_docs[idx]

                # ✅ PRIORITY 1: match numbers (years, values)
                if query_numbers:
                    if any(num in doc for num in query_numbers):
                        best_doc = doc
                        best_score = scores[idx].item()
                        break

                # ✅ PRIORITY 2: keyword overlap
                doc_words = set(re.findall(r'\w+', doc.lower()))
                overlap = query_words.intersection(doc_words)

                if len(overlap) >= 2:
                    best_doc = doc
                    best_score = scores[idx].item()
                    break

            # ✅ fallback
            if best_doc is None:
                idx = top_indices[0]
                best_doc = filtered_docs[idx]
                best_score = scores[idx].item()

            # -------------------------------
            # 🔥 FIX 2 ENDS HERE
            # -------------------------------

            return best_doc, best_score

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return None, 0.0