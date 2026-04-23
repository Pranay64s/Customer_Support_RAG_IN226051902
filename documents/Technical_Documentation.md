# Technical Documentation: RAG-Based Customer Support Assistant

## 1. Introduction
- **What is RAG:** Retrieval-Augmented Generation (RAG) is an architectural approach that combines the generative capabilities of Large Language Models (LLMs) with targeted information retrieval systems. It fetches relevant data from external sources and injects it into the LLM's prompt.
- **Why it is needed:** LLMs suffer from hallucinations and knowledge cut-offs. They don't inherently know private company data. RAG solves this by grounding the LLM's generation in factual, retrieved data.
- **Use case overview:** A Customer Support Assistant that reads a company's PDF manuals/policies. Users can ask questions, and the assistant retrieves the exact section to formulate an accurate answer. If the query is unanswerable or requires human judgment, the workflow pauses and escalates to a human agent.

## 2. System Architecture Explanation
The system follows a bipartite architecture: Data Ingestion and Inference/Workflow.
- **Data Ingestion:** A one-time or periodic process where PDFs are parsed, split into manageable chunks, embedded using a neural network, and stored in ChromaDB.
- **Inference/Workflow:** Handled by LangGraph. It is structured as a state machine. The state carries the user query, retrieved context, generated answer, and an escalation flag. The graph routes between a retrieval node, an LLM processing node, and a human-in-the-loop (HITL) node based on conditions evaluated dynamically.

## 3. Design Decisions
- **Chunk size choice:** A chunk size of `1000` characters with a `100` character overlap was selected. This provides a balance: large enough to capture the context of a paragraph or two, but small enough to maintain precise relevance during retrieval without maxing out token limits.
- **Embedding strategy:** Use standard pre-trained embeddings (like `SentenceTransformers` or OpenAI's text-embedding models) as they offer strong semantic search out of the box.
- **Retrieval approach:** Top-K semantic similarity search. We retrieve the top 4 chunks (`k=4`) to give the LLM sufficient context without overwhelming it with irrelevant data.
- **Prompt design logic:** The LLM is strictly instructed via the system prompt: "You are a customer support agent. Answer the user's question using ONLY the provided context. If the context does not contain the answer, set the escalate flag to true."

## 4. Workflow Explanation
- **LangGraph usage:** We use LangGraph instead of typical LangChain chains because it allows us to define cycles and conditional edges easily, which is crucial for HITL.
- **Node responsibilities:**
  - `retrieve`: Modifies the state by adding fetched chunks to the `context` field.
  - `process_llm`: Modifies the state by generating the `answer` and setting the `escalate` boolean.
  - `human_intervention`: Prompts for input from a human and overwrites the `answer`.
- **State transitions:** The flow dictates that every query goes through retrieval, then LLM generation. After generation, a conditional edge evaluates `escalate`.

## 5. Conditional Logic
- **Intent detection:** We use the LLM itself as a routing mechanism. By asking the LLM to output structured data (e.g., JSON or using tool calling/function calling), we can extract both its text answer and its boolean decision on whether escalation is needed.
- **Routing decisions:** 
  - `escalate == False` -> Transition to `END` -> Return answer to user.
  - `escalate == True` -> Transition to `human_node`.

## 6. HITL Implementation
- **Role of human intervention:** Acts as a fail-safe for queries that are too complex, ambiguous, or outside the scope of the knowledge base. It ensures customer satisfaction when the AI is uncertain.
- **Benefits and limitations:**
  - *Benefits:* High accuracy, avoids hallucination propagation, builds user trust.
  - *Limitations:* Slower response time when escalated, requires human bandwidth. We simulate this by pausing the python execution using LangGraph breakpoints or standard I/O waits.

## 7. Challenges & Trade-offs
- **Retrieval accuracy vs speed:** Increasing `k` improves accuracy but adds token cost and latency.
- **Chunk size vs context quality:** Smaller chunks ensure high retrieval relevance but might cut off sentences or lose surrounding context. The 100-character overlap mitigates this.
- **Cost vs performance:** Local embeddings (HuggingFace) and local LLMs save costs but might have slightly lower performance/reasoning capabilities compared to proprietary APIs like OpenAI.

## 8. Testing Strategy
- **Testing approach:** We test with a sample PDF (e.g., a sample policy document). 
- **Sample queries:** 
  - *Direct Match:* "What is the return policy?" (Should answer directly).
  - *Out of Scope:* "How do I build a nuclear reactor?" (Should escalate or gracefully decline).
  - *Ambiguous:* "It doesn't work, help." (Should escalate to human).

## 9. Future Enhancements
- **Multi-document support:** Allow ingestion of entire directories of varied formats (docx, txt).
- **Feedback loop:** Allow users to rate the answer, which could be logged to improve the chunking/retrieval strategy.
- **Memory integration:** Add conversation history so users can ask follow-up questions.
- **Deployment:** Wrap the application in a FastAPI backend and a React/Next.js frontend.
