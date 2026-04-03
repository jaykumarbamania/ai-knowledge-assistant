import os
import numpy as np
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = "data/faiss_index/index.bin"
DOC_PATH = "data/faiss_index/documents.pkl"

# ----------- LOAD DOCUMENTS (TXT + PDF) -----------

def load_documents():
    docs = []

    # Load TXT
    # if os.path.exists("data/knowledge.txt"):
    #     with open("data/knowledge.txt", "r") as f:
    #         docs.append(f.read())

    # Load PDFs
    upload_folder = "data/uploads"
    if os.path.exists(upload_folder):
        for file in os.listdir(upload_folder):
            if file.endswith(".pdf"):
                reader = PdfReader(os.path.join(upload_folder, file))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                docs.append(text)

    return "\n".join(docs)


# ----------- LOAD OR CREATE INDEX -----------

try:
    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_PATH):
        print("Loading existing index...")
        index = faiss.read_index(INDEX_PATH)

        with open(DOC_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        raise Exception("No index found")

except:
    print("Creating new index...")

    raw_text = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    documents = text_splitter.split_text(raw_text)

    embeddings = []

    for doc in documents:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        embeddings.append(response.data[0].embedding)

    embedding_matrix = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    os.makedirs("data/faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOC_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("Index created successfully.")

# ----------- QUERY LOOP -----------

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
                "content": "Answer using the provided context. Combine information if needed."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3
    )

    print("\nAI Answer:\n", response.choices[0].message.content)