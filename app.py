import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="PDF RAG Chat (Groq)", page_icon="📄")
st.title("📄 Chat with your PDF (Groq + RAG)")

# ---------------------------
# Sidebar: API key + settings
# ---------------------------
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        index=0,
    )
    chunk_size = st.number_input("Chunk size", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=150, step=50)
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)

# ---------------------------
# Helper functions (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def build_vectorstore(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def ask_groq(client, model, context, question):
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "context below. If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------
# Session state
# ---------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# File upload
# ---------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Reading and indexing PDF..."):
            text = extract_text(uploaded_file)
            if not text.strip():
                st.error("No extractable text found in this PDF.")
            else:
                st.session_state.vectorstore = build_vectorstore(
                    text, chunk_size, chunk_overlap
                )
                st.session_state.messages = []
                st.success("PDF processed! You can now ask questions below.")

# ---------------------------
# Chat interface
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the PDF...")

if question:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.error("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.vectorstore.similarity_search(
                    question, k=top_k
                )
                context = "\n\n".join(d.page_content for d in docs)
                client = Groq(api_key=groq_api_key)
                try:
                    answer = ask_groq(client, model_name, context, question)
                except Exception as e:
                    answer = f"Error calling Groq API: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
