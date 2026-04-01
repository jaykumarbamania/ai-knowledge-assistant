import os
import numpy as np
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = "data/faiss_index/index.bin"
DOC_PATH = "data/faiss_index/documents.pkl"

# Step 1: Check if index exists
if os.path.exists(INDEX_PATH) and os.path.exists(DOC_PATH):
    print("Loading existing FAISS index...")

    index = faiss.read_index(INDEX_PATH)

    with open(DOC_PATH, "rb") as f:
        documents = pickle.load(f)

else:
    print("Creating new FAISS index...")

    with open("data/knowledge.txt", "r") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    documents = text_splitter.split_text(text)

    embeddings = []

    for doc in documents:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        embeddings.append(response.data[0].embedding)

    embedding_matrix = np.array(embeddings).astype("float32")

    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    # Save index
    os.makedirs("data/faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOC_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("Index created and saved.")

# Step 2: Query loop
while True:
    question = input("\nAsk question (exit to quit): ")

    if question.lower() == "exit":
        break

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    query_vector = np.array([query_embedding]).astype("float32")

    k = 5
    distances, indices = index.search(query_vector, k)

    retrieved_chunks = [documents[i] for i in indices[0]]
    context = "\n---\n".join(retrieved_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Use the provided context and combine information if needed."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3
    )

    print("\nAI Answer:\n", response.choices[0].message.content)