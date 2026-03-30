import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Knowledge base
# documents = [
#     "Spring Boot is a Java framework used to build backend applications.",
#     "AWS provides cloud services like EC2, S3, and RDS.",
#     "Python is widely used in AI and machine learning.",
#     "Docker helps containerize applications for deployment.",
#     "Vector databases are used in AI systems for similarity search."
# ]

with open("data/knowledge.txt", "r") as f:
    text = f.read()

documents = text.split("\n\n")

# Generate embeddings
doc_embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    doc_embeddings.append(response.data[0].embedding)

embedding_matrix = np.array(doc_embeddings).astype("float32")

# Create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print("Knowledge base ready.")

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # Convert question to embedding
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    query_vector = np.array([query_embedding]).astype("float32")

    # Retrieve top documents
    k = 2
    distances, indices = index.search(query_vector, k)

    retrieved_docs = [documents[i] for i in indices[0]]

    context = "\n".join(retrieved_docs)

    # Ask LLM with context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer questions based only on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3
    )

    print("\nAI Answer:\n", response.choices[0].message.content)