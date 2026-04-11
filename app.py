import streamlit as st
from pipeline.detector import HallucinationDetector

# Load corpus
def load_corpus():
    with open("data/sample_corpus.txt") as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_detector():
    return HallucinationDetector()

st.set_page_config(page_title="LLM Hallucination Detector", layout="wide")

st.title("🔍 LLM Hallucination Detection System")

question = st.text_area("User Question")
response = st.text_area("LLM Response")

if st.button("Analyze"):

    detector = load_detector()
    result = detector.analyze(question, response)

    st.subheader("Overall Hallucination Score")
    st.metric("Score", round(result["overall_score"], 3))

    for idx, sent in enumerate(result["sentences"]):

        st.markdown("---")
        st.subheader(f"Sentence {idx+1}")

        st.write("**Text:**", sent["sentence"])

        if sent["label"] == "contradiction":
            st.error("❌ Contradiction Detected")
        elif sent["label"] == "entailment":
            st.success("✅ Supported by Evidence")
        elif sent["label"] == "not_verifiable":
            st.warning("⚠️ No Relevant Evidence Found")
        else:
            st.info("ℹ️ Neutral")

        st.write("**Hallucinated Spans:**", sent["hallucinated_spans"])
        st.write("**Score:**", round(sent["score"], 3))
        st.write("**Evidence:**", sent["evidence"])
