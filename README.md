# 📄 OpenAI RAG Document QA (Notebook-Only) — LlamaParse + ChromaDB + Evaluation

A **notebook-based Retrieval-Augmented Generation (RAG)** document question-answering pipeline that loads a PDF, retrieves the most relevant passages, and generates grounded answers using **OpenAI Chat models**. The notebook also includes **model comparison** (multiple LLM backends) and a lightweight **evaluation suite** (hit rate, MRR, faithfulness, relevancy, correctness + response time).

> ✅ **Note:** This project is intentionally **not deployed** (Notebook-only). It is designed for experimentation, benchmarking, and demonstrating RAG evaluation workflows.

---

## 🎯 What This Project Demonstrates

- End-to-end **RAG pipeline** for PDF document QA
- **Better PDF parsing** using **LlamaParse** (with fallback to PyPDFLoader)
- **Vector search** with **ChromaDB** (local, notebook-friendly)
- Multiple embedding strategies:
  - **OpenAIEmbeddings**
  - **BGE (BAAI/bge-large-en-v1.5)** via Hugging Face
- LLM comparisons across:
  - **OpenAI Chat models** (e.g., gpt-3.5 variants; optional GPT-4)
  - Optional open-source HF models (with fallback options depending on runtime)
- Evaluation + benchmarking utilities:
  - Response-time comparison
  - Retrieval/answer quality proxies: **Hit Rate, MRR, Faithfulness, Relevancy, Correctness**
  - Summary tables + radar chart visualization

---

## 🧠 Architecture Overview (RAG Flow)

1. **Load PDF**
   - Primary: **LlamaParse** (Markdown output for stronger extraction)
   - Fallback: **PyPDFLoader**

2. **Chunking**
   - RecursiveCharacterTextSplitter (`chunk_size=1000`, `chunk_overlap=200`)

3. **Embedding**
   - Baseline: OpenAI embeddings
   - Alternative: BGE-large embeddings (normalized)

4. **Vector Store**
   - **ChromaDB** (notebook/local-friendly; supports quick experimentation)

5. **Retrieval**
   - Top-k semantic retrieval (`k=5`)

6. **Answer Generation**
   - RetrievalQA chain with an enhanced prompt template (grounded, context-first)

7. **Evaluation**
   - Multi-question benchmarking across models and metrics

---

## 📌 Example Use Case in This Notebook

The notebook demonstrates document QA on an insurance policy PDF (LIC “New Jeevan Shanti”) and tests questions like:

- What is this policy and what does it offer?
- What are the annuity options?
- How is the death benefit calculated?
- What is the free look period?
- What are the tax implications?

---

## 🧰 Tech Stack

- **LangChain** – RAG orchestration (chunking, retrieval, chains)
- **OpenAI** – LLM generation + embeddings
- **ChromaDB** – vector database for retrieval (local / notebook)
- **LlamaParse** – improved PDF parsing (optional, recommended)
- **Hugging Face** – BGE embeddings + optional local LLM experiments
- **Python / Colab** – notebook execution environment

---

## 🔐 API Keys & Security

This repo is structured to avoid exposing secrets.

### Required environment variables
- `OPENAI_API_KEY` (required)
- `LLAMA_API_KEY` (optional, only needed if using LlamaParse)

✅ Recommended approach (Notebook / Colab):
- Use environment variables or Colab secrets
- Do **not** commit keys or credential files

Example (local `.env`, not committed):
```env
OPENAI_API_KEY=your_openai_key_here
LLAMA_API_KEY=your_llamaparse_key_here
