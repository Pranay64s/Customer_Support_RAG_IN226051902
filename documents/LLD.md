# Low-Level Design (LLD): RAG-Based Customer Support Assistant

## 1. Module-Level Design
- **Document Processing Module:** Handles the file I/O operations, ensuring the PDF is readable and properly parsed.
- **Chunking Module:** Utilizes `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=100`.
- **Embedding Module:** Wraps around an embedding model (e.g., `OpenAIEmbeddings` or `HuggingFaceEmbeddings`).
- **Vector Storage Module:** Interacts with the `Chroma` client, managing the collection, adding documents, and persisting the database.
- **Retrieval Module:** Configures the retriever with a specific `k` (e.g., top 4) and possibly a similarity score threshold.
- **Query Processing Module:** Formats the user query, structures the prompt template for the LLM.
- **Graph Execution Module:** Implements `StateGraph` from LangGraph. Compiles the nodes and edges.
- **HITL Module:** A specific node or a LangGraph interruption checkpoint that waits for an API call or console input from a human.

## 2. Data Structures
- **Document Representation:** `{ "page_content": "...", "metadata": { "source": "pdf", "page": 1 } }`
- **Chunk Format:** Same as Document, but constrained by character count.
- **Embedding Structure:** A float array representing the chunk's vector, e.g., `[0.12, -0.45, 0.88, ...]`.
- **Query-response schema:** `{ "query": "string", "response": "string", "confidence": float, "escalate": bool }`
- **State Object (for LangGraph):**
  ```python
  from typing import TypedDict, List
  class GraphState(TypedDict):
      query: str
      context: List[str]
      answer: str
      escalate: bool
      human_response: str
  ```

## 3. Workflow Design (LangGraph)
- **Nodes:**
  - `retrieve_node`: Fetches context from ChromaDB based on the query.
  - `process_node`: Passes the context and query to the LLM to generate an answer and determine if escalation is needed.
  - `human_node`: The HITL node that pauses execution to wait for a human answer.
- **Edges:**
  - `START -> retrieve_node`
  - `retrieve_node -> process_node`
  - `process_node -> conditional_edge`
    - If `escalate == False`, go to `END`.
    - If `escalate == True`, go to `human_node`.
  - `human_node -> END`
- **State Flow:** The `GraphState` dictionary is updated at each node and passed to the next.

## 4. Conditional Routing Logic
- **Answer Generation Criteria:** The LLM is instructed to answer the query solely based on the retrieved context.
- **Escalation Criteria:** 
  - **Low confidence:** If the retrieved chunks do not contain the answer, the LLM sets `escalate: True`.
  - **Missing context:** If the context is empty or irrelevant.
  - **Complex query:** Explicit user request for a human agent (e.g., "let me talk to a human").

## 5. HITL Design
- **When Escalation is Triggered:** The `process_node` determines `escalate=True`. LangGraph routing directs to `human_node`. LangGraph's `interrupt_before=["human_node"]` or similar pausing mechanism is used.
- **What Happens After Escalation:** The system state is saved (using a thread/memory saver in LangGraph). The user/system is notified that a human is needed.
- **How Human Response is Integrated:** The human provides input (e.g., via a CLI prompt or an API endpoint). The graph execution is resumed with the `human_response` state variable updated. The final answer becomes the human's response.

## 6. API / Interface Design
- **Input format (CLI):** `> Enter your query: How do I reset my password?`
- **Output format (CLI):** `> Assistant: You can reset your password by...` or `> Routing to human agent...`
- **Interaction flow:** Continuous while-loop for a CLI application.

## 7. Error Handling
- **Missing data:** If the PDF is not found or empty, the application halts gracefully during startup.
- **No relevant chunks found:** The retriever might return empty. The LLM gracefully states it cannot answer or triggers an escalation.
- **LLM failure:** Catch exceptions (e.g., API timeouts or rate limits) and return a standard "System unavailable, please try again" message.
