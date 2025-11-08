
from abc import ABC, abstractmethod

'''
Abstract base class for the classes that wrap the tools that the agent can use
'''
class AgentToolsBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def getToolDefinition(self):
        pass

    @abstractmethod
    def runFunctionIfNeeded(self, function_call):
        pass