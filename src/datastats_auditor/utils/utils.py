


import inspect
from functools import wraps

def capture_params(store_attr="_captured_params"):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            print(f"bound.kwargs: {bound.kwargs}")
            print(f"bound.args: {bound.args}")
            # Store parameters on the function object
            #setattr(wrapper, store_attr, dict(bound.arguments))
            
            setattr(wrapper, store_attr, dict(bound.kwargs))

            return func(*args, **kwargs)

        return wrapper
    return decorator