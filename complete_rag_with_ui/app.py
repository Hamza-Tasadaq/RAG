import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

DIMENSIONS = 1536

# Anchor to this script's location, then go up to the RAG root where the
# real chroma_langchain_db lives (this file is in RAG/complete_rag_with_ui/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "..", "chroma_langchain_db")
COLLECTION_NAME = "example_collection"

st.set_page_config(page_title="JS Notes RAG Chatbot", page_icon="📘")


# --- Cache expensive resources so they're only created once per session ---
@st.cache_resource
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=DIMENSIONS,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


@st.cache_resource
def load_agent():
    return create_agent(
        model="google_genai:models/gemini-flash-latest",
        system_prompt=SystemMessage(
            content=(
                "You are a helpful assistant. Answer the question based ONLY on the "
                "provided context. If the answer isn't in the context, say you don't know."
            )
        ),
    )


vector_store = load_vector_store()
agent = load_agent()

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    # st.caption(f"📦 Chunks in DB: {vector_store._collection.count()}")
    k = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=3)
    show_sources = st.checkbox("Show retrieved sources", value=True)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("📤 Upload a document")
    uploaded_file = st.file_uploader("Upload a PDF to add it to the knowledge base", type=["pdf"])

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if uploaded_file is not None:
        if uploaded_file.name in st.session_state.processed_files:
            st.info(f"'{uploaded_file.name}' has already been added this session.")
        else:
            if st.button(f"Process '{uploaded_file.name}'"):
                with st.spinner("Loading and chunking PDF..."):
                    # PyPDFLoader needs a real file path, so write to a temp file first
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        loader = PyPDFLoader(tmp_path)
                        new_docs = loader.load()

                        # Tag each page with the original filename for source display
                        for doc in new_docs:
                            doc.metadata["source"] = uploaded_file.name

                        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                        new_chunks = splitter.split_documents(new_docs)
                    finally:
                        os.remove(tmp_path)

                with st.spinner(f"Embedding {len(new_chunks)} chunks..."):
                    vector_store.add_documents(new_chunks)

                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"Added {len(new_chunks)} chunks from '{uploaded_file.name}' to the knowledge base.")
                st.rerun()

st.title("📘 RAG Chatbot")
st.caption("Ask questions about your JavaScript notes — answers are grounded in the PDF content.")

# --- Chat history state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render existing chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    page = src["metadata"].get("page", "N/A")
                    source = src["metadata"].get("source", "unknown")
                    st.markdown(f"**Chunk {i}** — `{source}`, page {page}\n\n{src['content']}")

# --- Chat input ---
prompt = st.chat_input("What is your question?")

if prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks..."):
            results = vector_store.similarity_search(query=prompt, k=k)

        context_text = "\n\n---\n\n".join(doc.page_content for doc in results)
        augmented_prompt = f"""Context:
{context_text}

Question: {prompt}"""

        with st.spinner("Generating answer..."):
            result = agent.invoke({"messages": HumanMessage(content=augmented_prompt)})
            answer = result["messages"][-1].content[0]["text"]

        st.markdown(answer)

        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]

        if show_sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources, 1):
                    page = src["metadata"].get("page", "N/A")
                    source = src["metadata"].get("source", "unknown")
                    st.markdown(f"**Chunk {i}** — `{source}`, page {page}\n\n{src['content']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )