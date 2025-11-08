from google.genai import types
from agent_tools_base import AgentToolsBase

'''
A set of coffee related tools, to be used by the agent
'''
class CoffeeTools(AgentToolsBase) :
    def __init__(self) -> None:
        pass

    def getToolDefinition(self):
        make_coffee_and_taste_function = {
            "name": "make_coffee_and_taste_it",
            "description": "Prepare an actual cup of coffee with the given tools and materials, using the given instructions, taste it and rank it."
            "The returned ranking is on a scale of 0-10, where 10 is perfect and 0 is undrinkable",
            "parameters": {
                "type": "object",
                "properties": {                    
                    "tools_and_materials": {
                        "type": "string",
                        "description": "The tools and materials used to prepare the cofee",
                    },                    
                    "instructions": {
                        "type": "string",
                        "description": "The instructions for how to make the cup of coffee, using the given tools_and_materials. Provide the instructions as you would say them to a professional barista.",
                    },
                },
                "required": ["tools_and_materials", "instructions" ],
            },
        }

        obtain_tools_and_materials_function = {
            "name": "obtain_tools_and_materials",
            "description": "Obtain the desired tools and materials. Everything is free of charge and available, all tools and materials are perfect.",
            "parameters": {
                "type": "object",
                "properties": {                    
                    "tools_and_materials": {
                        "type": "string",
                        "description": "The tools and materials to obtain",
                    },                                        
                },
                "required": ["tools_and_materials"],
            },
        }
        tools = [
            types.Tool(function_declarations=[make_coffee_and_taste_function]),
            types.Tool(function_declarations=[obtain_tools_and_materials_function])
        ]
        return tools

    def make_coffee_and_taste_it(self, tools_and_materials, instructions):
        print(f"Using {tools_and_materials} To make your coffee, following these instructions {instructions}")
        print(f"Tasting it...")        
        tasting_result = "Prepared according to instructions, and tasted. The ranking for this cup is 8 out of 10"
        print(tasting_result)
        return types.Part.from_function_response(name="make_coffee_and_taste_it",response={"result": tasting_result},)        

    def obtain_tools_and_materials(self, tools_and_materials):
        message = f"Got the {tools_and_materials} you wanted! All in perfect condition!"
        print(message)
        return types.Part.from_function_response(name="obtain_tools_and_materials",response={"result": message},)        

    def runFunctionIfNeeded(self, function_call):
        if function_call.name == "make_coffee_and_taste_it":
            return self.make_coffee_and_taste_it(
                tools_and_materials=function_call.args["tools_and_materials"], 
                instructions=function_call.args["instructions"])
        elif function_call.name == "obtain_tools_and_materials":                    
            return self.obtain_tools_and_materials(tools_and_materials=function_call.args["tools_and_materials"])
        
        return None
            