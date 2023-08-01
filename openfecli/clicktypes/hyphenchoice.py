import click

def _normalize_to_hyphen(string):
    return string.replace("_", "-")

class HyphenAwareChoice(click.Choice):
    def __init__(self, choices, case_sensitive=True):
        choices = [_normalize_to_hyphen(choice) for choice in choices]
        super().__init__(choices, case_sensitive)

    def convert(self, value, param, ctx):
        value = _normalize_to_hyphen(value)
        return super().convert(value, param, ctx)
