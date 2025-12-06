import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

# ============================================================
# 1. KONFIGURASI
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PASTE_KUNCI_GEMINI_ANDA_DI_SINI")
if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_KUNCI_GEMINI_ANDA_DI_SINI":
    st.error("Kunci API Gemini tidak ditemukan. Harap set GEMINI_API_KEY.")
    st.stop()

# Model ekstraksi cepat
llm_rag = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

# Model khusus JSON (lebih taat)
llm_json = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

# ============================================================
# 2. DEFINISI PYDANTIC
# ============================================================

class KanonikSummary(BaseModel):
    judul_dokumen: str
    poin_utama_1: str
    poin_utama_2: str
    poin_utama_3: str

class ValidationResult(BaseModel):
    is_valid: bool
    reasoning: str

# ============================================================
# 3. HELPER: Ekstrak JSON dari respon LLM
# ============================================================

def extract_json(text: str) -> str:
    """Mengambil blok JSON dari string menggunakan regex."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM tidak mengembalikan JSON valid:\n\n" + text)
    return match.group(0)

# ============================================================
# 4. VECTORSTORE
# ============================================================

@st.cache_resource
def process_pdf_and_create_vector_store(uploaded_file):
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)

    os.remove(temp_path)
    return vectorstore

# ============================================================
# 5. RINGKASAN KANONIK
# ============================================================

def generate_kanonik_summary(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    context_docs = retriever.invoke("Berikan poin-poin utama dan judul dokumen ini.")
    context = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = f"""
Anda adalah model AI yang WAJIB mengeluarkan JSON VALID tanpa teks lain.

FORMAT:
{{
    "judul_dokumen": "",
    "poin_utama_1": "",
    "poin_utama_2": "",
    "poin_utama_3": ""
}}

KONTEN DOKUMEN:
{context}

Keluarkan hanya JSON.
"""

    response = llm_json.invoke(prompt)
    cleaned = extract_json(response)
    data = json.loads(cleaned)

    return KanonikSummary(**data)

# ============================================================
# 6. VALIDASI TWEET
# ============================================================

def validate_tweet_content(kanonik_summary, user_tweet):
    kanonik_str = json.dumps(kanonik_summary.dict(), indent=2)

    prompt = f"""
Anda adalah validator konten. Jawab hanya dalam JSON valid.

FORMAT:
{{
    "is_valid": true/false,
    "reasoning": ""
}}

Ringkasan:
{kanonik_str}

Tweet:
{user_tweet}

Jawab hanya dengan JSON.
"""

    response = llm_json.invoke(prompt)
    cleaned = extract_json(response)
    data = json.loads(cleaned)

    return ValidationResult(**data)

# ============================================================
# 7. STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Validated Tweet App", layout="centered")
st.title("🐦 Validated Tweet App")
st.caption("RAG Extraction + JSON Validation • Gemini 2.5 Flash + Gemini 2.5 Pro")

with st.form("tweet_form"):
    uploaded_file = st.file_uploader("Unggah PDF", type="pdf")
    user_tweet = st.text_area("Tweet Anda:", height=140)
    submitted = st.form_submit_button("Validasi Tweet")

if submitted:
    try:
        if not uploaded_file or not user_tweet:
            st.error("Unggah PDF dan isi tweet terlebih dahulu.")
            st.stop()

        with st.spinner("📄 Memproses PDF..."):
            vectorstore = process_pdf_and_create_vector_store(uploaded_file)

        with st.spinner("📝 Menghasilkan Ringkasan Kanonik..."):
            kanonik = generate_kanonik_summary(vectorstore)
            st.subheader("✅ Ringkasan Kanonik")
            st.json(kanonik.dict())

        with st.spinner("🔍 Memvalidasi Tweet..."):
            validation = validate_tweet_content(kanonik, user_tweet)

        st.subheader("📌 Hasil Validasi")

        if validation.is_valid:
            st.success("✅ Tweet VALID dan konsisten dengan dokumen!")
            st.balloons()
        else:
            st.error("❌ Tweet TIDAK konsisten dengan dokumen.")

        st.markdown(f"**Alasan:** {validation.reasoning}")

    except Exception as e:
        st.error("Terjadi error saat memproses.")
        st.exception(e)
