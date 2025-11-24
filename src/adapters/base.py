# src/adapters/base.py
from abc import ABC, abstractmethod

class BaseEnvAdapter(ABC):
    """
    Abstract base class for benchmark adapters.
    All specific environments (Alfworld, WebShop) must implement these methods.
    """
    
    @abstractmethod
    def reset(self) -> str:
        """Reset environment and return initial task description (str)"""
        pass

    @abstractmethod
    def step(self, action_command: str) -> str:
        """
        Execute action.
        Returns:
            str: Formatted observation (including success/failure markers)
        """
        pass

    @abstractmethod
    def get_tool_config(self) -> dict:
        """
        Return configuration dictionary for AgentScope Toolkit registration.
        Contains: function, function_name, function_desc
        """
        pass
