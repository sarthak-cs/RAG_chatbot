from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-pro")

# 1. Load your document
loader = TextLoader("data/data.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 3. Embeddings + Vector Database
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, emb)

# --- Chat loop ---
print("RAG Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    results = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Build prompt
    prompt = f"""
    Use ONLY the following context to answer the question:

    {context}

    Question: {query}
    """

    # LLM Response
    response = model.generate_content(prompt)

    answer = response.text.strip()
    print("\nBot:", answer, "\n")
