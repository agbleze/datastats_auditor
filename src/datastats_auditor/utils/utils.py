import inspect
import importlib, pkgutil
from datastats_auditor.stats import (drift, image_stats,
                                     object_stats,
                                     split_service
                                     )
from datastats_auditor.stats import datacard
import datastats_auditor.io as io
import inspect
from functools import wraps


MODULES = [drift,
           image_stats, 
           object_stats, 
           split_service,
           io, datacard
           ]

def capture_params(store_attr="_captured_params"):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            print(f"bound.kwargs: {bound.kwargs}")
            print(f"bound.args: {bound.args}")
            
            setattr(wrapper, store_attr, dict(bound.kwargs))

            return func(*args, **kwargs)

        return wrapper
    return decorator


def get_cls_init_params(cls):
    sig = inspect.signature(cls.__init__)
    return [p.name for p in sig.parameters.values() if p.name != "self"]


def discover_plugins():
    for module in MODULES:
        for _, modname, _ in pkgutil.iter_modules(module.__path__):
            importlib.import_module(f"{module.__name__}.{modname}")


