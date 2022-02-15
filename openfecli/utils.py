import importlib

def import_thing(import_string):
    splitted = import_string.split('.')
    if len(splitted) > 1:
        obj = splitted[-1]
        mod = ".".join(splitted[:-1])
        module = importlib.import_module(mod)
        result = getattr(module, obj)
    else:
        mod = splitted[0]
        result = importlib.import_module(mod)
    return result
