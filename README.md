# 🧠 LLM Debugging Q&A with LangSmith

This project demonstrates an end-to-end pipeline for evaluating LLM responses using LangSmith, LangChain, and OpenAI. It includes:

- Creation of a LangSmith dataset containing common debugging Q&A for LangChain and LLM development.
- Retrieval-Augmented Generation (RAG) system that fetches context from LangSmith documentation.
- Generation of answers using OpenAI's GPT-4o model.
- Custom evaluation using LangSmith's `evaluate()` function, including a length-based evaluator to check if the output is concise compared to the reference.

---

### 💡 Project Workflow

1. **Dataset Creation**  
   Upload a curated Q&A dataset to LangSmith using the Python SDK.

2. **Vectorstore Retrieval**  
   Load and embed LangSmith documentation using `RecursiveCharacterTextSplitter` and OpenAI embeddings. Persist locally using SKLearnVectorStore.

3. **RAG Response Generation**  
   Query relevant context from the vectorstore, then use GPT-4o to generate an answer based on retrieved content.

4. **Custom Evaluation**  
   Run LangSmith’s `evaluate()` to compare generated outputs against dataset references using a custom metric (`is_concise_enough`).

---

### ⚙Setup

1. Install dependencies:

```bash
pip install -r requirements.txt


### set environment variables
```bash
export OPENAI_API_KEY=your-openai-key
export LANGSMITH_API_KEY=your-langsmith-key


