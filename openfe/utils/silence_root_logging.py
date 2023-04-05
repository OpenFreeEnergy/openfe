import contextlib
import logging

@contextlib.contextmanager
def silence_root_logging():
    """Context manager to silence logging from root logging handlers.

    a.k.a, "Why are you using basicConfig during import -- or in library
    code at all?"
    """
    root = logging.getLogger()
    old_handlers = root.handlers
    for handler in old_handlers:
        root.removeHandler(handler)

    null = logging.NullHandler()
    root.addHandler(null)
    yield
    root.removeHandler(null)
    for handler in old_handlers:
        root.addHandler(handler)



