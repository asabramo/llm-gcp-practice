from firestore import FirestoreMan
from embedding import EmbeddingMan
from llm import LlmMan
import xmltodict

import firestore 

def ingestDB(fs):
    print("Ingesting")
    input_file_name = "./Data/coffee.stackexchange.com/posts.xml"
    # xml = ET.parse(input_file_name)
    # root_element = xml.getroot()

    # for child in root_element:
    #     for field in child:
    #         print(f"Field {field}")
    # obj = untangle.parse(input_file_name)
    # for row in obj["posts"]:
    #     print(f"row {row}")
    #     for field in row:
    #         print(f"field {field}")
    #         exit()

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
fs = FirestoreMan();
#ingestDB(fs)

#fs.fetch("1");
input_query="What happens if you add salt to your coffee?"
embedder = EmbeddingMan()
embedded_query = embedder.calcEmbedding(input_query)
docs = fs.fetchKnn(vector=embedded_query, k = 3);
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

llm = LlmMan()
# response = llm.makeLlmQuery(input_query)
# print(f"LLM Response without the RAG is:\n{response}")

prompt = llm.buildLlmPrompt(base_prompt="You are a cofee guru, answer the questions of the cofee enthusiasts",
    related_docs=docs,
    query=input_query)
print(f"Prompt: {prompt}")
response = llm.makeLlmQuery(prompt)
print(f"LLM Response WITH RAG is:\n{response}")


# val = {
#     "id": "100",
#     "name": "Hundred"
# }
# fs.put(key = "MyKey1", value=val)
# fs.fetch("MyKey1");


