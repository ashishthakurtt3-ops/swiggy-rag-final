import os
import streamlit as st

st.set_page_config(
    page_title="Swiggy Annual Report Q&A",
    page_icon="🍊",
    layout="wide"
)

st.markdown("""
<style>
    .answer-box {
        background-color: #fff4ec;
        border-left: 4px solid #FC8019;
        padding: 15px 20px;
        border-radius: 6px;
        font-size: 15px;
        line-height: 1.7;
        margin-top: 10px;
    }
    .stButton>button {
        background-color: #FC8019;
        color: white;
        border: none;
        border-radius: 6px;
    }
    .stButton>button:hover {
        background-color: #e06010;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍊 Swiggy Annual Report Q&A")
st.caption("Ask anything about Swiggy's FY 2023-24 Annual Report. All answers come strictly from the document.")

PDF_PATH = "Annual-Report-FY-2023-24.pdf"
INDEX_DIR = "index"

if "question" not in st.session_state:
    st.session_state.question = ""
if "answer" not in st.session_state:
    st.session_state.answer = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False


@st.cache_resource(show_spinner=False)
def build_and_load_pipeline():
    from rag.document_processor import DocumentProcessor
    from rag.embeddings import EmbeddingModel
    from rag.vector_store import VectorStore
    from rag.retriever import Retriever
    from rag.generator import AnswerGenerator

    embedder = EmbeddingModel()

    if os.path.isdir(INDEX_DIR):
        store = VectorStore.load(INDEX_DIR)
    elif os.path.isfile(PDF_PATH):
        processor = DocumentProcessor()
        chunks = processor.process(PDF_PATH)
        embeddings = embedder.embed_texts([c.text for c in chunks])
        store = VectorStore(dimension=embedder.dimension)
        store.add_chunks(chunks, embeddings)
        store.save(INDEX_DIR)
    else:
        return None, None

    retriever = Retriever(embedder, store, top_k=5)
    generator = AnswerGenerator()
    return retriever, generator


with st.spinner("Loading... please wait (this may take a minute on first run)"):
    retriever, generator = build_and_load_pipeline()

if retriever is None:
    st.error("Annual Report PDF not found.")
    st.stop()

st.success("Ready! Ask your question below.")
st.divider()

st.markdown("**Quick questions:**")
samples = [
    "What is Swiggy's total revenue?",
    "Who are the board of directors?",
    "What is Swiggy Instamart?",
    "What are the key risks?",
    "What is Swiggy's net loss?",
]

cols = st.columns(len(samples))
for i, q in enumerate(samples):
    if cols[i].button(q, use_container_width=True):
        st.session_state.question = q
        st.session_state.auto_run = True

st.divider()

question = st.text_input(
    "Your question:",
    value=st.session_state.question,
    placeholder="e.g. What is Swiggy's Adjusted EBITDA for FY 2023-24?"
)
show_sources = st.checkbox("Show source sections from the report", value=False)

run_answer = st.button("Get Answer", type="primary") or st.session_state.auto_run

if run_answer and question.strip():
    st.session_state.auto_run = False
    with st.spinner("Searching the report and generating answer..."):
        try:
            context, sources = retriever.get_context(question)
            answer = generator.generate(question, context)
            st.session_state.answer = answer
            st.session_state.sources = sources
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            st.info("Please try again.")

if st.session_state.answer:
    st.markdown("### Answer")
    st.markdown(f'<div class="answer-box">{st.session_state.answer}</div>', unsafe_allow_html=True)

    if st.session_state.sources:
        pages = list(set(str(s["page"]) for s in st.session_state.sources))
        st.markdown(f"**Sources: Page(s) {', '.join(pages)}**")

        if show_sources:
            with st.expander("View relevant sections from the report"):
                for i, s in enumerate(st.session_state.sources, 1):
                    st.markdown(f"**Section {i} — Page {s['page']} (relevance: {s['score']})**")
                    st.text(s["text"])
                    st.divider()