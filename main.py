import xmltodict
from agent import AiAgent
from embedding import EmbeddingMan
from coffee_tools import CoffeeTools
from rag_tools import RagTools

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

system_prompt = """You are a cofee guru, answer the questions of the cofee enthusiasts, 
                if they ask you to DO something for them, you are more than happy to do it, using the functions at your disposal. 
                Assume you have no materials and tools, you need to obtain them to have them. 
                Always start by looking up expert information, if you cannot find any, only then apply common sense.
                Your goal is to make the customer the best cup of coffee for them.                
                If you believe the goal has been reached, write GOAL_REACHED_YAY in the response 
                """

print("Hi, I'm your Cofee Guru, I'd love to make you the best possible cup of coffee. What would you like?")
print("""Classic examples:
        - Which beans and espresso machine do I need for a classic Italian afternoon espresso? Make me one of those
        - "Make me a cup of cofee using Illy ground coffee and a Belliani machinetta, buy anything you need to"
      """)
input_query=input()
embedder = EmbeddingMan()
rag_tools = RagTools(embedder=embedder)
coffee_tools = CoffeeTools()
agent = AiAgent(agent_tools=[rag_tools, coffee_tools])
agent.runAgent(system_prompt=system_prompt, input_query=input_query)

print(f"Your request was: {input_query}, we hope it has been fully resolved.")


