from llm import LlmMan

'''
A generic agent class, wrapping Gemini.
The agent does its best to fulfill your wishes, using the tools you provide it
'''
class AiAgent:
    def __init__(self, agent_tools=[]) -> None:
        self.agentState = { 
            "iterations": [], 
        }
        self.goalReached = False
        self.maxIterations = 7
        self.GOAL_MARKER = "GOAL_REACHED_YAY"
        self.iteration = 0 
        self.agent_tools = agent_tools   
        tools = []
        for agent_tool in self.agent_tools:
            tools += agent_tool.getToolDefinition()
        self.llm = LlmMan(tools=tools)
        print("Agent Initialized!")
        print(f"Available tools:")

        for tool in tools:
            for function in tool.function_declarations:
                print(f"- {function.name}")
        pass


    '''
    Run the agent, loop until the goal is reached
    The flow:
        while !goalReached
            Increment iteration
            Prepare current iteration prompt and data based on last iteration results
            Call the LLM - (Ask the LLM to say GOAL_REACHED when it thinks the goal is reached)
            Process the response
            Check if goal is reached
                BREAK
            check for tool requests
                Run requested tools
                Add tool requests results to the state
            Push LLM response + function responses to state    
    '''
    def runAgent(self, system_prompt, input_query, pause_every_step=False):       
        
        prompt = self.llm.buildLlmPrompt(base_prompt=system_prompt, query=input_query)
    
        # The main agent loop - continue until the goal is reached
        while not self.goalReached and self.iteration < self.maxIterations:
            # Init this iteration's state
            self.iteration += 1
            print(f"**************** ITERATION {self.iteration} ******************")
            lastLlmResponseContent = []
            lastToolResponses = []
            if self.iteration > 1:
                # Update the prompt to include the state and results of tool executions
                lastIterationData = self.agentState["iterations"][-1];
                lastLlmResponseContent = lastIterationData["llm_response_content"]
                lastToolResponses = lastIterationData["tool_responses"] if "tool_responses" in lastIterationData else []
                base_prompt = """Continue to help the user, look at the state to know what you've already done. 
                                 If there's nothing more to do, make sure to include the marker """ + f"{self.GOAL_MARKER} in the response."
                prompt = self.llm.buildLlmPrompt(base_prompt=base_prompt, query=input_query)                
                
            thisIterationState = { 
                "iteration": self.iteration,
                "prompt": prompt                    
            }
            self.agentState['iterations'].append(thisIterationState)
            response = self.llm.makeLlmQuery(prompt, state=lastLlmResponseContent, tool_responses=lastToolResponses)
            # Find candidate
            candidate = self.findCandidate(response)            
            
            thisIterationState["llm_response_content"] = candidate.content
            print(self.getTextAnswer(candidate=candidate))

            # Check if the goal is reached, the loop will not continue if it's reached
            self.checkGoal(candidate=candidate)
            
            self.runToolsIfNeeded(thisIterationState=thisIterationState, candidate=candidate)                        
            
            if pause_every_step and (not self.goalReached):
                response = input("Continue to next iteration? (y): ")
                if response != 'y':
                    print("abort per user request")
                    self.dumpState()
                    break
            

    def findCandidate(self, response):
        # Check for abnormal termination
        if len(response.candidates) == 0:
                print("The LLM has nothing to say, exiting");
                self.dumpAndExit()
            
        if response.candidates[0].content == None:
            print("The LLM returned a candidate with no content");
            self.dumpAndExit()
        
        if response.candidates[0].content.parts == None:
            print("The LLM returned content with no Parts");
            self.dumpAndExit()
        # Ignoring multiple candidates, always take the first candidate
        return response.candidates[0]
    
    # Check if the goal has been reached
    def checkGoal(self, candidate):
        for part in candidate.content.parts:
            if part.text and self.GOAL_MARKER in part.text:
                print("GOAL REACHED!!!")
                self.goalReached = True
                break;

    def getTextAnswer(self, candidate):
        text = ""
        for part in candidate.content.parts:
            if part.text:
                text += part.text
        return text


    # Returns true if any tools were found
    def runToolsIfNeeded(self, thisIterationState, candidate):
        # Check for a function call
        functions_to_call = []
        for part in candidate.content.parts:
            if part.function_call:
                functions_to_call.append(part.function_call)
                # print(f"Found function call in part: {part}")
        if len(functions_to_call) > 0:
            #TODO: extend to more than one function            
            function_call = functions_to_call[0]
            print(f"Function to call: {function_call.name}")
            print(f"Arguments: {function_call.args}")
            tool_responses = []
            for agent_tool in self.agent_tools:
                tool_resp = agent_tool.runFunctionIfNeeded(function_call=function_call)
                if tool_resp:
                    tool_responses.append(tool_resp)                
            thisIterationState['tool_responses'] = tool_responses
        return 'tool_responses' in thisIterationState
            
    
    def dumpAndExit(self):
        self.dumpState()
        exit()

    def dumpState(self):
        #print("<<<<<<<<<<<<<<< AGENT STATE >>>>>>>>>>>>>>>>")
        #print(f"{self.agentState}")
        with open("agentDump.txt", "w") as f:
            f.write(str(self.agentState))
        