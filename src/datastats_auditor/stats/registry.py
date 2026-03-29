class Registry:
    
    def __init__(self):
        self._registry = {}
        
    def register(self, name: str, cls: type, status:str):
        if name in self._registry:
            raise ValueError(f"{name} is already registered.")
        
        self._registry[name] = {name: cls, "status": status}
        
    def get(self, name: str, status: str):
        if name not in self._registry.keys():
            raise ValueError(f"{name} is not in registry.")
        obj = self._registry[name]
        if status == obj["status"]:
            return obj[name]
        else:
            raise ValueError(f"{name} is registered as {obj['status']}, not {status}.")
    
    def list_available(self, status = None):
        if not status:
            return set(self._registry.keys())
        else:
            return set([name for name, obj in self._registry.items() if obj["status"] == status])


registry = Registry()       