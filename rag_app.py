import os
import sys
from typing import TypedDict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# --- State Definition ---
class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    escalate: bool
    human_response: str

# --- Configuration ---
PDF_PATH = "RAG_Project_Complete.docx.pdf"
CHROMA_DB_DIR = "./chroma_db"

# Global retriever to avoid reloading PDF every time
retriever_instance = None

def get_retriever():
    global retriever_instance
    if retriever_instance is not None:
        return retriever_instance

    print("Setting up RAG pipeline (this might take a moment on the first run)...")
    if not os.path.exists(PDF_PATH):
        print(f"Error: Could not find {PDF_PATH}. Please ensure it is in the same directory.")
        sys.exit(1)

    # 1. Load PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. Embedding and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    
    retriever_instance = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever_instance

def get_llm():
    # If no GOOGLE_API_KEY is present, we'll use a mocked response for demonstration.
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nWARNING: GOOGLE_API_KEY not found. Using a mock LLM for demonstration purposes.")
        class MockStructured:
            def __init__(self, schema):
                self.schema = schema
            def invoke(self, inputs):
                prompt = str(inputs)
                escalate = "human" in prompt.lower() or "agent" in prompt.lower() or "help" in prompt.lower()
                answer = "I am a mock LLM. I retrieved the context but since there is no API key, I am returning this mock answer."
                if escalate:
                    answer = "I am not sure, let me escalate."
                return self.schema(answer=answer, escalate=escalate)

        class MockLLM:
            def with_structured_output(self, schema):
                return MockStructured(schema)
        return MockLLM()
    
    # Initialize the real LLM
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class LLMOutput(BaseModel):
    answer: str = Field(description="The answer to the user's query based ONLY on the context.")
    escalate: bool = Field(description="True if the context does not contain the answer, or if the user requests a human, else False.")

# --- Nodes ---
def retrieve_node(state: GraphState):
    print("-> Retrieving context...")
    query = state["query"]
    retriever = get_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context}

def process_node(state: GraphState):
    print("-> Processing with LLM...")
    query = state["query"]
    context = state["context"]
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(LLMOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a customer support agent. Answer the question using ONLY the context provided below. If you cannot answer it from the context, or if the user asks for a human, set escalate to True.\n\nContext:\n{context}"),
        ("human", "{query}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"context": context, "query": query})
    
    return {"answer": result.answer, "escalate": result.escalate}

def human_node(state: GraphState):
    print("\n--- HUMAN IN THE LOOP TRIGGERED ---")
    print(f"The LLM could not confidently answer the query or the user asked for a human.")
    print(f"User Query: {state['query']}")
    print("-----------------------------------")
    human_input = input("Please provide the answer as a human agent: ")
    return {"human_response": human_input, "answer": human_input}

# --- Edges ---
def should_escalate(state: GraphState):
    if state["escalate"]:
        return "human_node"
    return END

# --- Build Graph ---
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("process", process_node)
    workflow.add_node("human_node", human_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "process")
    workflow.add_conditional_edges("process", should_escalate)
    workflow.add_edge("human_node", END)
    
    app = workflow.compile()
    return app

def main():
    print("=========================================")
    print(" RAG-Based Customer Support Assistant")
    print("=========================================")
    print("Note: To use real LLM generation, set the GOOGLE_API_KEY environment variable.")
    print("Example: set GOOGLE_API_KEY=AIzaSy...\n")
    
    app = build_graph()
    
    while True:
        try:
            user_query = input("\nEnter your query (or type 'quit' to exit): ")
            if user_query.lower() in ['quit', 'exit']:
                break
                
            initial_state = {"query": user_query, "context": "", "answer": "", "escalate": False, "human_response": ""}
            
            # Run graph
            result = app.invoke(initial_state)
            
            print("\n=== FINAL RESPONSE ===")
            print(result["answer"])
            print("======================\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
