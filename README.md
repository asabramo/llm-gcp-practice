# LLM Practice on GCP

The data is from: https://archive.org/details/stackexchange
What this repo does:
- Ingests data from stack exchange public data
- Creates embeddings for it using Gemini-embedding
- Pushes the records and the embeddings to Firestore
- Take a user query, sends it to gemini and asks it to answer it, based on the relevant document from Firestore, as received by a KNN search
So far, it's a classic RAG

# TODO:
- Convert the RAG fetch to be a tool, so that the LLM can decide whether and when it wants to use it
- Implement an agent wrapper, so that it can 

#Setup (windows)
```bat
prepare.bat
python 
```