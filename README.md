
# 🧠 AI Knowledge Assistant (RAG AI System)

An AI-powered Knowledge Assistant built using **Retrieval-Augmented Generation (RAG)** architecture.  
This system allows users to query information from their own documents (TXT/PDF) using semantic search and LLMs.

---

## 🚀 Features

- 💬 Chat with your own data
- 📄 Supports TXT and PDF document ingestion
- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware answers using LLM (OpenAI)
- 💾 Persistent FAISS vector database (no recomputation)
- ⚡ Fast retrieval and response generation
- 🛠 Fault-tolerant index loading (auto rebuild on failure)

---

## 🧠 System Architecture

```

Documents (TXT / PDF)
↓
Text Extraction
↓
Chunking (with overlap)
↓
Embeddings (OpenAI)
↓
FAISS Vector Database
↓
User Query
↓
Similarity Search
↓
Context + Query → LLM
↓
Final Answer

```

---

## 🛠 Tech Stack

- **Language:** Python
- **LLM:** OpenAI (gpt-4o-mini)
- **Embeddings:** text-embedding-3-small
- **Vector DB:** FAISS

### Libraries:
- numpy
- faiss-cpu
- python-dotenv
- langchain-text-splitters
- pypdf

---

## 📁 Project Structure

```

ai-knowledge-assistant/
│
├── app.py
├── rag_test.py
├── rag_with_chunking.py
├── rag_persistent.py
├── rag_with_upload.py
│
├── data/
│   ├── knowledge.txt
│   ├── uploads/            # PDF files
│   └── faiss_index/
│       ├── index.bin
│       └── documents.pkl
│
├── .env
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ai-knowledge-assistant
````

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add OpenAI API Key

Create `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Running the Project

### Basic RAG (Static Data)

```bash
python rag_test.py
```

---

### With Chunking

```bash
python rag_with_chunking.py
```

---

### Persistent Vector DB (Recommended)

```bash
python rag_persistent.py
```

---

### With PDF Upload Support

```bash
python rag_with_upload.py
```

---

## 📄 How to Add Documents

### TXT:

Add content in:

```
data/knowledge.txt
```

### PDF:

Place files in:

```
data/uploads/
```

---

## 🧠 Key Concepts Implemented

* LLM Chat Completion
* Prompt Engineering
* Embeddings & Semantic Search
* Vector Databases (FAISS)
* Retrieval-Augmented Generation (RAG)
* Chunking with Overlap
* Persistent Storage
* Fault Tolerance

---

## 🔥 Challenges Solved

* Multi-chunk retrieval issue → fixed via top-k + prompt tuning
* FAISS index corruption → handled with auto-rebuild
* Context merging → improved prompt engineering

---

## 💼 Resume Description

Built an AI-powered knowledge assistant using Retrieval-Augmented Generation (RAG) with OpenAI APIs, FAISS vector database, and document ingestion (TXT/PDF), including chunking, semantic search, and persistent indexing for optimized performance.

---

## 🚀 Future Enhancements

* 🌐 FastAPI backend
* 🎨 Frontend UI (React)
* ☁️ AWS Deployment (ECS + S3)
* 🔐 Authentication & multi-user support
* 📊 Monitoring & logging
* 🧠 Hybrid search (keyword + vector)

---

## 👨‍💻 Author

Jay Bamania
