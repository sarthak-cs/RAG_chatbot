# RAG Chatbot using Gemini 2.5 + LangChain + ChromaDB

This repository contains a Retrieval-Augmented Generation (RAG) chatbot built using:

- Google Gemini 2.5 Pro
- LangChain
- HuggingFace Embeddings
- Chroma Vector Store

It loads a custom dataset, splits it into chunks, embeds them, and retrieves relevant context for every query.

---

## 🚀 Features
- Document ingestion (txt file)
- Recursive text chunking
- Chroma vector database
- HuggingFace MiniLM embeddings
- Gemini 2.5 Pro for smart answers
- Console-based interactive chatbot

---

---

## ▶️ Run the Chatbot

pip install -r requirements.txt
python src/rag_chatbot.py


---

## 🔑 .env Format

GEMINI_API_KEY=your_api_key_here

---
