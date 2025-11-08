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

## Step 3: Make it an Agent
- Generalized the tools class
- Create an Agent class that loops until it reaches the goal, making tool calls and LLM calls as needed
- The agent is generic, it knows nothing about coffee, the main function provides coffee related tools and prompts

# TODO:
- Use native Gemini infras for calling python functions and for goal detection
- Add better user interface

# Setup (windows)
```bat
prepare.bat
python 
```