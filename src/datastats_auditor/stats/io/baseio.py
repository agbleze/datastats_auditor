from abc import ABC, abstractmethod
from typing import Iterable, Literal
import pandas as pd
from ..registry import registry

class IterableDataset(ABC):
    name: str = "base_image_importer"
    status: Literal["stable", "experimental"] = "stable"
    required = ["name", "status"]
    valid_status_values = ["stable", "experimental"]
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr in cls.required:
            if attr not in cls.__dict__:
                raise NotImplementedError(f"{cls.__name__} requires definining a '{attr}' class attribute because it is a Subclass of IterableDataset.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")
        registry.register(cls.name, cls, cls.status)
        
    @abstractmethod
    def __iter__(self, *args, **kwargs) -> Iterable:
        raise NotImplementedError("Subclasses must implement the __iter__ method.")
    
         
class BaseAnnotationDFImporter(ABC):
    name: str = "base_annotation_df_importer"
    status: Literal["stable", "experimental"] = "stable"
    required = ["name", "status"]
    valid_status_values = ["stable", "experimental"]
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr in cls.required:
            if attr not in cls.__dict__:
                raise NotImplementedError(f"{cls.__name__} requires definining a '{attr}' class attribute because it is a Subclass of BaseAnnotationDFImporter.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")

        registry.register(cls.name, cls, cls.status)
        
    @abstractmethod
    def load(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the load method.")