# LLM Practice on GCP

The data is from: https://archive.org/details/stackexchange
What this repo does:
- Ingests data from stack exchange public data
- Creates embeddings for it using Gemini-embedding
- Pushes the records and the embeddings to Firestore
- Take a user query, sends it to gemini and asks it to answer it, based on the relevant document from Firestore, as received by a KNN search
## Step 1 So far, it's a classic RAG

## Step 2: Introduce tools
- Convert the RAG fetch to be a tool, and let the LLM decide whether and when it wants to use it, although we are strongly nudging it towards that

# TODO:
- Implement an agent wrapper. The goal would be to make the best coffee. We'll have a class that implements coffee tools like "buy equipment", "Make Coffee", and "Taste", and the agent will work with the RAG and action tools until it reaches a good enough result.

# Setup (windows)
```bat
prepare.bat
python 
```