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
        prompt = base_prompt + "\n" + "Here is the user's query. You may request information from the tools you are provided. For each available tool, explain why you choose to call it or not. Answer percisely and concisely, only what was asked, but make sure to quote the source of the information when applicable" + query
        if related_docs != "":
            prompt = prompt + "\nHere are documents related to the topic, if they answer the question you are asked, use them and quote which one you used:\n" + \
            str(related_docs)
        return prompt

    def makeLlmQuery(self, prompt):
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(
                tools=self.tools,
                temperature=0,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
                max_output_tokens=100,
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
        print(response)
        return response


