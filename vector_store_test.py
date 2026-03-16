import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Sample documents
documents = [
    "Spring Boot is a Java framework used to build backend applications.",
    "AWS provides cloud services like EC2, S3, and RDS.",
    "Python is widely used in AI and machine learning.",
    "Docker helps containerize applications for deployment.",
    "Vector databases are used in AI systems for similarity search."
]

# Step 2: Generate embeddings for all documents
embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

# Convert to numpy array
embedding_matrix = np.array(embeddings).astype("float32")

# Step 3: Create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embedding_matrix)

print("Stored documents in FAISS index:", index.ntotal)

# Step 4: Query
query = "What is used to deploy applications in containers?"

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

query_vector = np.array([query_embedding]).astype("float32")

# Step 5: Search similar vectors
k = 2  # number of nearest results
distances, indices = index.search(query_vector, k)

print("\nTop matching documents:")
for i in indices[0]:
    print("-", documents[i])