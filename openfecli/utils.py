import importlib

def import_thing(import_string):
    splitted = import_string.split('.')
    obj = splitted[-1]
    mod = ".".join(splitted[:-1])
    module = importlib.import_module(mod)
    return getattr(module, obj)
