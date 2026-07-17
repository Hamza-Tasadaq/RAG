from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage

DIMENSIONS=1536

load_dotenv()  # reads variables from .env into environment


agent = create_agent(
    model="google_genai:models/gemini-flash-latest",
    system_prompt=SystemMessage(
        content="You are a helpful assistant. Answer the question based ONLY on the "
                "provided context. If the answer isn't in the context, say you don't know."
    ),
)


prompt=input("Whats your question?")
# --- Embed ---

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=DIMENSIONS,
)


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # saves to disk
)


results = vector_store.similarity_search(
    query=prompt,
    k=3,  # number of similar documents to retrieve
)

print(f"Retrieved {len(results)} similar documents from Chroma")



# --- Build context text from retrieved chunks ---
context_text = "\n\n---\n\n".join(doc.page_content for doc in results)


# --- Combine context + question into one message ---
augmented_prompt = f"""Context:
{context_text}

Question: {prompt}"""

result = agent.invoke({"messages": HumanMessage(content=augmented_prompt)})

print(result["messages"][-1].content[0]["text"])
