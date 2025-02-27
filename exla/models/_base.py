"""
Base model class for all models in the EXLA SDK.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModel(ABC):
    """
    Base class for all models in the EXLA SDK.
    """
    
    def __init__(self):
        """Initialize the base model."""
        self.name = "BaseModel"
        self.device = "cpu"
    
    @abstractmethod
    def _install_dependencies(self) -> None:
        """
        Install required dependencies for the model.
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model.
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def inference(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run inference with the model.
        This method should be implemented by subclasses.
        
        Returns:
            Dictionary containing inference results.
        """
        return {"status": "not_implemented"} 