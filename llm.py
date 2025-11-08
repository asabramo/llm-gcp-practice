import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


class LlmMan:
    def __init__(self, tools):
        load_dotenv()
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.tools = tools
    
    def buildLlmPrompt(self, base_prompt, query, related_docs=""):
        prompt = base_prompt + \
        """
        You may request information from the tools you are provided - to call a tool, make sure to pass the proper function part, not only say you want to call it in the text.
        For each available tool, explain why you choose to call it or not.
        Answer percisely and concisely, only what was asked, but make sure to quote the source of the information when applicable.
        Here is the user's query:
        """ + query
        if related_docs != "":
            prompt = prompt + "\nHere are documents related to the topic, if they answer the question you are asked, use them and quote which one you used:\n" + \
            str(related_docs)
        return prompt

    def makeLlmQuery(self, prompt, state=[], tool_responses=[]):

        # print(f"makeLlmQuery: Prompt: {prompt} \n tool_responses: {tool_responses} \n")

        contents = [types.Part.from_text(text=prompt)];
        if state != []:
            contents.append(types.Part.from_text(text="The state of our conversation so far: " + str(state)))
        if tool_responses != []:
            contents.append(types.Content(role="user", parts=tool_responses))
            
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(
                tools=self.tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                temperature=0,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
                max_output_tokens=500,
                stop_sequences=['STOP!'],
                presence_penalty=0.0,
                frequency_penalty=0.0,
                safety_settings=[
                   types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_ONLY_HIGH',
                    )
                ]
            ),            
        )   
        # print(response)
        return response


