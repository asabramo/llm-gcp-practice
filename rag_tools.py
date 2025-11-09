from agent_tools_base import AgentToolsBase
from firestore import FirestoreMan
from google.genai import types


'''
A RAG tool, answering coffee related questions, to be used by the agent
TODO: generalize so that the RAG and the coffee related stuff are de-coupled
'''
class RagTools(AgentToolsBase):
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.fs = FirestoreMan()

    def getToolDefinition(self):
        fetch_related_docs_function = {
            "name": "coffee_expert_search",
            "description": "Coffee expert answers for any coffee related questions",
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
        return tools


    def fetchRelatedDocs(self, input_query, k = 3):
        embedded_query = self.embedder.calcEmbedding(input_query)
        docs = self.fs.fetchKnn(vector=embedded_query, k=k);
        for doc in docs:
            del doc['embedding_field']
        print(f"Input Query: {input_query}")        
        print(f"Related docs:")        
        for doc in docs:
            # print(f"{doc}")
            if "@Title" in doc:
                print(f"Title: {doc["@Title"]}")
            else:
                print("No Title")
            if "@Body" in doc:
                print(f"Body: {doc["@Body"]}")
            else:
                print("No Body")
        return types.Part.from_function_response(name="coffee_expert_search",response={"result": docs},)        

    def runFunctionIfNeeded(self, function_call):
        if function_call.name == "coffee_expert_search":
            return self.fetchRelatedDocs(input_query=function_call.args["query"])                

        return None
    