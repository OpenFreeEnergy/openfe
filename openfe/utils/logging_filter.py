import logging

class MsgIncludesStringFilter:
    def __init__(self, string):
        self.string = string

    def filter(self, record):
        return not self.string in record.msg
