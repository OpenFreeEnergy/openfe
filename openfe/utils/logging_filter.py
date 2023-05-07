import logging

class MsgIncludesStringFilter:
    """Logging filter to silence specfic log messages.

    See https://docs.python.org/3/library/logging.html#filter-objects

    Parameters
    ----------
    string : str
        if an exact for this is included in the log message, the log record
        is suppressed
    """
    def __init__(self, string):
        self.string = string

    def filter(self, record):
        return not self.string in record.msg
