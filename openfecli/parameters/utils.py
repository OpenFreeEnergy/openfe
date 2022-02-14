from plugcli.params import NOT_PARSED


def import_parameter(import_str):
    try:
        result = import_thing(user_input)
    except (ImportError, AttributeError):
        result = NOT_PARSED
    return result
