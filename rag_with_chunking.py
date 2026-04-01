import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load file
with open("data/knowledge.txt", "r") as f:
    text = f.read()

# Step 1: Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

documents = text_splitter.split_text(text)

print("Total chunks:", len(documents))

# Step 2: Create embeddings
embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

embedding_matrix = np.array(embeddings).astype("float32")

# Step 3: FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print("Vector DB ready.")

# Step 4: Query loop
while True:
    question = input("\nAsk question (exit to quit): ")

    if question.lower() == "exit":
        break

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    query_vector = np.array([query_embedding]).astype("float32")

    k = 3
    distances, indices = index.search(query_vector, k)

    retrieved_chunks = [documents[i] for i in indices[0]]
    context = "\n---\n".join(retrieved_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                # "content": "Answer only from the provided context. If not found, say 'I don't know'."
                "content": "You are an AI assistant. Use the provided context to answer. If multiple pieces of context are relevant, combine them to give a complete answer."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3
    )

    print("\nAI Answer:\n", response.choices[0].message.content)