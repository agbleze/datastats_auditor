from abc import ABC, abstractmethod
from typing import Literal
from ..registry import registry

class BaseCardCreator(ABC):
    name: str = "base_card_creator"
    status: Literal["stable", "experimental"] = "stable"
    required = ["name", "status"]
    valid_status_values = ["stable", "experimental"]
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr in cls.required:
            if attr not in cls.__dict__:
                raise NotImplementedError(f"{cls.__name__} requires definining a '{attr}' class attribute because it is a Subclass of BaseCardCreator.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")

        registry.register(cls.name, cls, cls.status)
        
    @abstractmethod
    def create_card(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the create_card method.")

    
    


