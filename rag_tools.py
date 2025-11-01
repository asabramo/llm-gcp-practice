from firestore import FirestoreMan
from google.genai import types


class RagTools:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.fs = FirestoreMan()

    def getToolDefinition(self):
        fetch_related_docs_function = {
            "name": "coffe-expert-search",
            "description": "Cofee expert answers for any coffee related questions",
            "parameters": {
                "type": "object",
                "properties": {                    
                    "query": {
                        "type": "string",
                        "description": "The term for which to fetch the related documents",
                    },                    
                    "k": {
                        "type": "number",
                        "description": "The number of related documents to fetch, default is 3",
                    },
                },
                "required": ["query"],
            },
        }
        tools = [types.Tool(function_declarations=[fetch_related_docs_function])]
        return tools;


    def fetchRelatedDocs(self, input_query, k = 3):
        embedded_query = self.embedder.calcEmbedding(input_query)
        docs = self.fs.fetchKnn(vector=embedded_query, k=k);
        print(f"Input Query: {input_query}")        
        return docs