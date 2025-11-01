from firestore import FirestoreMan
from embedding import EmbeddingMan
from llm import LlmMan
import xmltodict

import firestore
from rag_tools import RagTools
import rag_tools 

def ingestDB(fs):
    print("Ingesting")
    input_file_name = "./Data/coffee.stackexchange.com/posts.xml"
    
    # Open the file and read the contents
    with open(input_file_name, 'r', encoding='utf-8') as file:
        my_xml = file.read()

    # Use xmltodict to parse and convert 
    # the XML document
    my_dict = xmltodict.parse(my_xml)
    posts = my_dict["posts"]
    # print(posts)
    c = 0    
    rows = posts['row']    
    for post in rows:
        print(f"Uploading Post: {post}")
        embedder = EmbeddingMan()
        embedding = embedder.calcEmbedding(post)
        post["embedding_field"] = embedding;
        fs.put(post["@Id"], post)
        c += 1
        print(f"Uploaded Post number: {c}")
        if (c == 10000):
            print(f"Exited, c = {c} ")
            exit()
            


print("Starting!")
# fs = FirestoreMan();
#ingestDB(fs)

#fs.fetch("1");
input_query="Which coffee beans should I buy if I want to consume less caffeine?"
embedder = EmbeddingMan()

rag_tools = RagTools(embedder=embedder)
llm = LlmMan(tools=rag_tools.getToolDefinition())

# response = llm.makeLlmQuery(input_query)
# print(f"LLM Response without the RAG is:\n{response}")
base_prompt = "You are a cofee guru, answer the questions of the cofee enthusiasts"
prompt = llm.buildLlmPrompt(base_prompt=base_prompt, query=input_query)
print(f"Prompt: {prompt}")
response = llm.makeLlmQuery(prompt)
# Check for a function call
functions_to_call = [];
for candidate in response.candidates:
    for part in candidate.content.parts:
        if part.function_call:
            functions_to_call.append(part.function_call)
            print(f"Found function call in part: {part}")

if len(functions_to_call) > 0:
    #TODO: extend to more than one function
    function_call = functions_to_call[0]
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    #TODO: Generalize function calling with some kind of registry
    related_docs = rag_tools.fetchRelatedDocs(input_query=function_call.args["query"])
    print(f"Related docs:")        
    for doc in related_docs:
            # print(f"{doc}")
            if "@Title" in doc:
                print(f"Title: {doc["@Title"]}")
            else:
                print("No Title")
            if "@Body" in doc:
                print(f"Body: {doc["@Body"]}")
            else:
                print("No Body")
    #TODO: generalize into a loop
    prompt = llm.buildLlmPrompt(base_prompt=base_prompt, related_docs=related_docs, query=input_query)
    response = llm.makeLlmQuery(prompt)
else:
    print("No function call found in the response.")
    print(response.text)

print(f"Final LLM Response:\n{response}")


# val = {
#     "id": "100",
#     "name": "Hundred"
# }
# fs.put(key = "MyKey1", value=val)
# fs.fetch("MyKey1");


