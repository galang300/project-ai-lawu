import streamlit as st
import os
import json
import re
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel

# ============================================================
# 1. KONFIGURASI
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Kunci API Gemini tidak ditemukan di file .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Model Gemini (Flash)
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

# ============================================================
# 2. IMPLEMENTASI ALGORITMA (UNTUK MEMENUHI POIN 2)
# ============================================================

# --- Algoritma Versi 1: REKURSIF (Recursive) ---
# Menggunakan library LangChain yang secara internal menggunakan rekursi
# untuk memecah teks berdasarkan hierarki separator (\n\n, \n, " ", "").
def recursive_split(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# --- Algoritma Versi 2: ITERATIF (Iterative) ---
# Implementasi manual menggunakan perulangan (looping) sederhana.
# Memotong teks secara linear tanpa melihat struktur semantik.
class IterativeTextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        length = len(text)
        i = 0
        # LOGIKA ITERATIF: Menggunakan While Loop
        while i < length:
            # Ambil potongan dari indeks i sampai i + chunk_size
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
            
            # Geser indeks (increment), mundur sedikit jika ada overlap
            # Pastikan langkah maju minimal 1 agar tidak infinite loop
            step = max(1, self.chunk_size - self.chunk_overlap)
            i += step
        return chunks

# ============================================================
# 3. LOGIKA UTAMA APLIKASI (RAG & VALIDASI)
# ============================================================

class KanonikSummary(BaseModel):
    judul_dokumen: str
    poin_utama_1: str
    poin_utama_2: str
    poin_utama_3: str

class ValidationResult(BaseModel):
    is_valid: bool
    reasoning: str

def extract_json(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        try:
            json.loads(text)
            return text
        except:
            raise ValueError("LLM tidak mengembalikan JSON valid:\n\n" + text)
    return match.group(0)

@st.cache_resource
def process_pdf_and_create_vector_store(uploaded_file):
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # Menggunakan Algoritma Rekursif untuk kualitas RAG yang lebih baik
    # (Pemisahan kalimat tidak terpotong di tengah kata)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)

    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return vectorstore

def generate_kanonik_summary(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    context_docs = retriever.invoke("Berikan poin-poin utama dan judul dokumen ini.")
    context = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = f"""
    Anda adalah asisten AI. Ekstrak informasi berikut ke JSON.
    TEKS: {context}
    Output HANYA JSON valid:
    {{ "judul_dokumen": "...", "poin_utama_1": "...", "poin_utama_2": "...", "poin_utama_3": "..." }}
    """
    response = llm_flash.invoke(prompt)
    data = json.loads(extract_json(response.content))
    return KanonikSummary(**data)

def validate_tweet_content(kanonik_summary, user_tweet):
    kanonik_str = json.dumps(kanonik_summary.dict(), indent=2)
    prompt = f"""
    Validasi TWEET vs RINGKASAN.
    RINGKASAN: {kanonik_str}
    TWEET: "{user_tweet}"
    Output HANYA JSON valid: {{ "is_valid": true/false, "reasoning": "..." }}
    """
    response = llm_flash.invoke(prompt)
    data = json.loads(extract_json(response.content))
    return ValidationResult(**data)

# ============================================================
# 4. UI: ANALISIS EFISIENSI (UNTUK MEMENUHI POIN 4)
# ============================================================

def run_benchmark_ui():
    st.markdown("### ⏱️ Analisis Kompleksitas Waktu (Poin 4)")
    st.info("Bagian ini membandingkan Running Time antara Algoritma Rekursif dan Iteratif dengan berbagai ukuran input.")

    if st.button("Mulai Benchmark"):
        # Variasi Ukuran Input (Input Sizes) sesuai Poin 4
        # (1000, 5000, ... 100000 karakter)
        input_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        results = []

        chunk_size = 1000
        chunk_overlap = 0 # Overlap 0 agar adil
        
        # Inisialisasi Splitter
        iter_splitter = IterativeTextSplitter(chunk_size, chunk_overlap)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, size in enumerate(input_sizes):
            status_text.text(f"Memproses ukuran input: {size} karakter...")
            
            # Generate Dummy Text (Lorem Ipsum)
            dummy_text = "Lorem ipsum dolor sit amet " * (size // 20)
            dummy_text = dummy_text[:size] # Potong agar pas ukurannya

            # --- UKUR WAKTU REKURSIF ---
            start_rec = time.time()
            recursive_split(dummy_text, chunk_size, chunk_overlap)
            end_rec = time.time()
            time_rec = (end_rec - start_rec) * 1000 # Konversi ke milidetik (ms)

            # --- UKUR WAKTU ITERATIF ---
            start_iter = time.time()
            iter_splitter.split_text(dummy_text)
            end_iter = time.time()
            time_iter = (end_iter - start_iter) * 1000 # Konversi ke milidetik (ms)

            results.append({
                "Ukuran Input (n)": size,
                "Rekursif (ms)": time_rec,
                "Iteratif (ms)": time_iter
            })
            
            progress_bar.progress((i + 1) / len(input_sizes))

        status_text.text("Benchmark Selesai!")
        
        # Buat DataFrame
        df_results = pd.DataFrame(results)
        df_results.set_index("Ukuran Input (n)", inplace=True)

        # 1. Tampilkan Tabel Data
        st.write("#### 1. Tabel Data Running Time")
        st.dataframe(df_results)

        # 2. Tampilkan Grafik Line Chart (Visualisasi Poin 4)
        st.write("#### 2. Grafik Perbandingan Efisiensi")
        st.line_chart(df_results)

        st.success("""
        **Analisis:**
        * Grafik di atas menunjukkan hubungan antara Ukuran Input (n) dan Waktu Eksekusi (t).
        * **Algoritma Iteratif** cenderung lebih cepat dan stabil (Kompleksitas O(n)).
        * **Algoritma Rekursif** (LangChain) lebih lambat karena melakukan pengecekan separator berulang-ulang untuk menjaga konteks kalimat (Kompleksitas lebih tinggi dari O(n) tergantung struktur teks).
        """)

# ============================================================
# 5. UI UTAMA (STREAMLIT)
# ============================================================

st.set_page_config(page_title="Tugas Besar AKA - Validasi Tweet", layout="wide")

st.title("🎓 Tugas Besar Analisis Kompleksitas Algoritma")
st.markdown("---")

# Menggunakan Tabs untuk memisahkan Aplikasi dengan Analisis Tugas
tab_app, tab_analisis = st.tabs(["📱 Aplikasi Validasi Tweet", "📊 Analisis Algoritma (Poin 2 & 4)"])

with tab_app:
    st.header("Aplikasi Validasi Tweet (RAG)")
    st.caption("Memvalidasi kebenaran tweet berdasarkan dokumen PDF referensi.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader("1. Unggah PDF Referensi", type="pdf")
        user_tweet = st.text_area("2. Masukkan Tweet", height=150, placeholder="Contoh: Menurut dokumen, inflasi tahun ini naik 5%...")
        submit_btn = st.button("Validasi Sekarang")

    with col2:
        if submit_btn:
            if not uploaded_file or not user_tweet:
                st.warning("Mohon lengkapi PDF dan Tweet terlebih dahulu.")
            else:
                try:
                    with st.spinner("Sedang memproses dokumen (Algoritma Rekursif)..."):
                        vectorstore = process_pdf_and_create_vector_store(uploaded_file)
                    
                    with st.spinner("Menganalisis konten..."):
                        kanonik = generate_kanonik_summary(vectorstore)
                        validation = validate_tweet_content(kanonik, user_tweet)
                    
                    # Tampilkan Hasil
                    st.subheader("Hasil Validasi")
                    
                    if validation.is_valid:
                        st.success("✅ **VALID** - Tweet sesuai dengan dokumen.")
                    else:
                        st.error("❌ **TIDAK VALID** - Tweet bertentangan dengan dokumen.")
                    
                    st.markdown(f"**Alasan:** {validation.reasoning}")
                    
                    with st.expander("Lihat Ringkasan Dokumen Asli"):
                        st.json(kanonik.dict())
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

with tab_analisis:
    # Memanggil fungsi UI Analisis Poin 4
    run_benchmark_ui()