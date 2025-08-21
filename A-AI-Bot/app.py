import os
import io
import gc
import json
import pdfplumber
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# ---- Optional: local retrieval (no external service)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# ENV + CLIENT INITIALIZE
# =========================
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(page_title="PDF Q&A (Groq)", layout="wide", page_icon="📄")

# =========================
# STYLES
# =========================
st.markdown(
    """
    <style>
      .app-container { background-color: #f6f7fb; padding: 0.5rem 1rem 1.25rem; border-radius: 12px; }
      .chunk-chip { display:inline-block; padding:4px 8px; margin: 2px; border-radius: 999px; background:#eef0ff; font-size:12px; }
      .citation { font-size: 0.9rem; color:#555; }
      .stTextArea textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
      .small { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def require_api_key():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Add it to a `.env` file like:\n\n`GROQ_API_KEY=your_key_here`")
        st.stop()

def read_pdf_pages(file_like) -> list[dict]:
    """
    Return a list of dicts: [{"page_num": 1-based, "text": "..."}]
    """
    pages = []
    with pdfplumber.open(file_like) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Clean up whitespace a bit
            text = " ".join(text.split())
            if text.strip():
                pages.append({"page_num": i, "text": text})
    return pages

@st.cache_data(show_spinner=False)
def build_tfidf_index(pages: list[dict]):
    """
    Build a TF-IDF index over page texts.
    Returns (vectorizer, matrix, page_texts).
    """
    page_texts = [p["text"] for p in pages]
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    matrix = vectorizer.fit_transform(page_texts)
    return vectorizer, matrix, page_texts

def retrieve(query: str, vectorizer, matrix, pages: list[dict], top_k: int = 4):
    """
    Return top_k page snippets and their metadata sorted by similarity.
    """
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).ravel()
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for idx in idxs:
        results.append({
            "page_num": pages[idx]["page_num"],
            "text": pages[idx]["text"],
            "score": float(sims[idx]),
        })
    return results

def summarize_long_text(text: str, limit: int = 1500) -> str:
    """
    Keep prompt under a rough char limit to avoid over-long contexts.
    """
    if len(text) <= limit:
        return text
    # Try to keep sentence boundaries roughly
    return text[:limit] + " ..."

def make_prompt(snippets: list[dict], question: str) -> str:
    """
    Build the model prompt with citations and instructions to answer from context.
    """
    context_blocks = []
    for s in snippets:
        header = f"[Source: Page {s['page_num']}]"
        context_blocks.append(f"{header}\n{s['text']}\n")
    context_text = "\n\n".join(context_blocks)
    system_rules = (
        "Answer the user's question **only** using the provided sources. "
        "Cite page numbers like (p. 2) after facts. If the answer is not in the sources, say you don't have enough info."
    )
    prompt = f"{system_rules}\n\nSources:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    return prompt

def call_groq(model: str, prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error from Groq: {e}"

def pretty_citations(snippets: list[dict]) -> str:
    if not snippets:
        return ""
    pages = [f"p.{s['page_num']}" for s in snippets]
    unique = []
    for p in pages:
        if p not in unique:
            unique.append(p)
    return "Sources: " + ", ".join(unique)

def download_chat_button(history):
    if not history:
        return
    payload = json.dumps(history, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download chat (JSON)",
        data=payload,
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )

# =========================
# SIDEBAR
# =========================
st.sidebar.title("PDF Query Assistant")
st.sidebar.caption("Built with Streamlit + Groq")

with st.sidebar.expander(" Settings", expanded=True):
    model = st.selectbox(
        "Groq model",
        options=["llama3-8b-8192", "llama3-70b-8192"],
        index=0,
        help="Larger models can be more accurate but slower/costlier."
    )
    top_k = st.slider("Context pages to use (Top‑K)", min_value=1, max_value=8, value=4, help="How many relevant pages to include as context.")
    show_extracted = st.toggle("Show extracted page text", value=False)

st.sidebar.markdown("---")
clear = st.sidebar.button("🧹 Clear Chat", use_container_width=True)

# =========================
# STATE
# =========================
if "pages" not in st.session_state:
    st.session_state.pages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "history" not in st.session_state:
    st.session_state.history = []

if clear:
    st.session_state.history = []

# =========================
# MAIN
# =========================
st.title(" PDF Q&A — RAG‑Lite with Groq")
st.markdown("<div class='app-container'>", unsafe_allow_html=True)
st.write("Upload a PDF, ask a question, and get an answer with page‑level citations.")

require_api_key()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    # Guard: very large PDFs → memory
    uploaded_file.seek(0, io.SEEK_END)
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    if size_mb > 50:
        st.warning("This PDF is quite large (>50MB). Extraction may be slow.")
    with st.spinner("Extracting pages..."):
        st.session_state.pages = read_pdf_pages(uploaded_file)

    if not st.session_state.pages:
        st.error("No readable text found in this PDF. Try another file or OCR.")
    else:
        st.success(f"Loaded {len(st.session_state.pages)} page(s).")
        with st.spinner("Building search index..."):
            st.session_state.index = build_tfidf_index(st.session_state.pages)
        st.toast("Index ready ", icon="✅")

        if show_extracted:
            with st.expander(" Extracted text by page"):
                for p in st.session_state.pages[:50]:  # avoid huge dumps
                    st.markdown(f"**Page {p['page_num']}**")
                    st.text_area(f"Page {p['page_num']} text", value=p["text"], height=150, key=f"text_{p['page_num']}")

question = st.text_input(" Ask a question about the PDF", placeholder="e.g., What are the key findings?")

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button(" Get Answer", type="primary", use_container_width=True)
with col2:
    download_chat_button(st.session_state.history)

if ask:
    if not uploaded_file or not st.session_state.pages:
        st.error("Please upload a PDF first.")
    elif not question.strip():
        st.error("Please type a question.")
    else:
        vectorizer, matrix, _ = st.session_state.index
        with st.spinner("Searching relevant pages..."):
            hits = retrieve(question, vectorizer, matrix, st.session_state.pages, top_k=top_k)

        if not hits:
            st.info("Couldn't find relevant content in the PDF for this query.")
        else:
            # Build prompt with only the best snippets (trim to keep prompt small)
            trimmed = [{"page_num": h["page_num"], "text": summarize_long_text(h["text"]), "score": h["score"]} for h in hits]
            prompt = make_prompt(trimmed, question)

            with st.spinner("Asking Groq..."):
                answer = call_groq(model, prompt)

            # Display
            st.markdown("###  Answer")
            st.write(answer)
            st.markdown(f"<span class='citation'>{pretty_citations(trimmed)}</span>", unsafe_allow_html=True)

            # Show which pages were used
            st.markdown("### 📚 Context used")
            chips = " ".join([f"<span class='chunk-chip'>Page {h['page_num']} • {h['score']:.2f}</span>" for h in hits])
            st.markdown(chips, unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "citations": [h["page_num"] for h in hits]
            })

            gc.collect()

# Show history
if st.session_state.history:
    st.markdown("### Chat History")
    for i, item in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"**Q{i}:** {item['question']}")
        st.markdown(f"**A{i}:** {item['answer']}")
        st.markdown(f"<span class='citation'>Pages: {', '.join('p.' + str(p) for p in item['citations'])}</span>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Use larger Top‑K if your document is long or your answers feel incomplete.")
st.sidebar.caption("© PDF Q&A — Streamlit + Groq")
