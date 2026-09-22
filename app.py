import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF, then ask questions about it. Powered by Groq.")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs locally/CPU
GROQ_MODEL = "llama-3.3-70b-versatile"      # change to any model your Groq key supports
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # characters overlapped between chunks
TOP_K = 4                # number of chunks retrieved per question

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Add it under Settings > Secrets.")
        st.stop()
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def extract_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index(chunks: list[str], embedder: SentenceTransformer):
    embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via normalized vectors
    index.add(embeddings)
    return index


def retrieve(question: str, embedder: SentenceTransformer, index, chunks: list[str], k: int = TOP_K) -> list[str]:
    q_emb = embedder.encode([question], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0] if i != -1]


def ask_groq(client: Groq, question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant answering questions about a PDF document. "
        "Use ONLY the provided context to answer. If the answer is not in the "
        "context, say you don't know based on the document."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------------------------------------------------------------------------
# Sidebar: upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Reading and indexing PDF..."):
            embedder = load_embedder()
            raw_text = extract_text(uploaded_file)

            if not raw_text.strip():
                st.error("No extractable text found in this PDF (it may be scanned/image-only).")
            else:
                chunks = chunk_text(raw_text)
                index = build_index(chunks, embedder)

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []
                st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'")

    if st.session_state.pdf_name:
        st.info(f"Active document: **{st.session_state.pdf_name}**")
        if st.button("Clear document"):
            st.session_state.chunks = None
            st.session_state.index = None
            st.session_state.pdf_name = None
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main: chat
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.index is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    question = st.chat_input("Ask something about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embedder = load_embedder()
                client = get_groq_client()
                relevant_chunks = retrieve(
                    question, embedder, st.session_state.index, st.session_state.chunks
                )
                answer = ask_groq(client, question, relevant_chunks)
                st.markdown(answer)

                with st.expander("Sources used"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Chunk {i}:** {chunk[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
