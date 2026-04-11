import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")

try:
    import streamlit
    print("✅ streamlit imported")
except ImportError as e:
    print(f"❌ streamlit failed: {e}")

try:
    import spacy
    print("✅ spacy imported")
except ImportError as e:
    print(f"❌ spacy failed: {e}")

try:
    import torch
    print("✅ torch imported")
except ImportError as e:
    print(f"❌ torch failed: {e}")

try:
    import faiss
    print("✅ faiss imported")
except ImportError as e:
    print(f"❌ faiss failed: {e}")

try:
    from pipeline.detector import HallucinationDetector
    print("✅ pipeline.detector imported")
except ImportError as e:
    print(f"❌ pipeline.detector failed: {e}")

try:
    from config.settings import TOP_K
    print(f"✅ config.settings imported (TOP_K: {TOP_K})")
except ImportError as e:
    print(f"❌ config.settings failed: {e}")
