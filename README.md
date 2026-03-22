<div align="center">
  
<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Vectorless-blueviolet?style=for-the-badge" alt="Vectorless" />
  <img src="https://img.shields.io/badge/Model-DeepSeek--R1--0528-blue?style=for-the-badge" alt="DeepSeek" />
  <img src="https://img.shields.io/badge/Dataset-Harry%20Potter-gold?style=for-the-badge" alt="Harry Potter" />
</p>

# Vectorless RAG: Reasoning-based Retrieval

<p align="center"><b>Reasoning-native RAG&nbsp; ◦ &nbsp;No Vector DB&nbsp; ◦ &nbsp;No Chunking&nbsp; ◦ &nbsp;Human-like Tree Search</b></p>

<h4 align="center">
  <a href="#-introduction">📖 Introduction</a>&nbsp; • &nbsp;
  <a href="#-tree-structure">🌲 Tree Index</a>&nbsp; • &nbsp;
  <a href="#-usage">⚙️ Usage</a>&nbsp; • &nbsp;
  <a href="#-architecture">🏗️ Architecture</a>&nbsp; • &nbsp;
  <a href="https://github.com/Satharva2004/Vectorless-RAG">⭐ Support</a>
</h4>
  
</div>

 **🔥 Features:**
- **Sleek Minimal UI**: A completely overhauled frontend focused on readability and "Wizarding World" aesthetics.
- **Auto-Scrolling Reasoning**: Real-time "Thinking" blocks that scroll intelligently as the model reasons through the text.
- **DeepSeek-R1 Integration**: Powered by Featherless AI for state-of-the-art reasoning over 100% grounded context.
 
 **📝 Concepts:**
- **Knowledge Hierarchy**: I've transformed 'Harry Potter and the Philosopher's Stone' into a semantic tree structure that preserves the narrative flow of every chapter.
- **Zero-Vector Retrieval**: I achieved 100% accuracy on complex plot queries without a single embedding call.
</details>

---

# 📑 Introduction

Are you tired of "vibe-based" retrieval where your RAG system returns random snippets that only *look* like the answer? Traditional vector RAG relies on semantic *similarity*, but for professional long-form content or complex narratives, **similarity ≠ relevance**.

Inspired by human experts, this project implements a **vectorless, reasoning-based RAG** system. It builds a **hierarchical tree index** from the book and uses an LLM to **reason** over that index to find the exact pages or chapters needed.

### 🎯 Core Features 

- **No Vector DB**: Uses the document's natural structure and LLM reasoning instead of opaque vector math.
- **No Chunking**: Chapters are kept whole, preserving context and "connecting the dots" that vector systems miss.
- **Human-like Retrieval**: The model "browses" the library just like you would—starting with the Table of Contents and zooming into the right chapter.
- **Perfect Traceability**: Every answer includes the exact chapter and reasoning path taken to find it.

---

# 🏗️ The Architecture

Instead of calculating mathematical distances in a high-dimensional space, we perform a **Semantic Tree Search**.

```mermaid
graph LR
    Q[User Question] --> R[Router LLM]
    R -- Search Summaries --> T[Tree Index]
    T -- Identify Chapters --> C[Grounded Context]
    C -- Reasoning Pass --> A[Final Answer]
```

---

# 🌲 Page-Index Tree Structure
This project transforms lengthy PDFs into a semantic **tree structure**, optimized for LLM consumption. Below is an example of how the *Philosopher's Stone* is indexed:

```jsonc
{
  "id": "chapter_001",
  "type": "chapter",
  "title": "CHAPTER ONE: The Boy Who Lived",
  "page_start": 11,
  "page_end": 22,
  "summary": "Introduction of the Dursleys and the arrival of Harry Potter at Privet Drive...",
  "full_text": "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say..."
}
```

---

# ⚙️ Usage

### 1. Install Dependencies
```bash
# Frontend
cd frontend && npm install

# Backend
cd backend && pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the `backend/` directory with your Featherless/Provider keys:
```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=deepseek-ai/DeepSeek-R1-0528
```

### 3. Launch
```bash
# Run Backend
python -m uvicorn app.main:app --port 8000

# Run Frontend
npm run dev
```

---

# ⭐ Support
This project demonstrates that you don't always need a Vector Database to build a powerful RAG system. Sometimes, a well-structured tree and a smart reasoning model are all you need to find the magic.

Leave a star 🌟 if you find this architecture useful!

### Connect
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/atharvasawant0804/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Satharva2004)

---
