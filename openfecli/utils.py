# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
import importlib
import functools
from typing import Callable, Optional
from datetime import datetime
import logging


def import_thing(import_string: str):
    """Obtain an object from a valid qualname (or fully qualified name)

    Parameters
    ----------
    import_string : str
        the qualname

    Returns
    -------
    Any :
        the object from that namespace
    """
    splitted = import_string.split('.')
    if len(splitted) > 1:
        # if the string has a dot, import the module and getattr the object
        obj = splitted[-1]
        mod = ".".join(splitted[:-1])
        module = importlib.import_module(mod)
        result = getattr(module, obj)
    else:
        # if there is no dot, import and return the module
        mod = splitted[0]
        result = importlib.import_module(mod)
    return result


def write(string: str):
    """

    This is abstracted so that we can change output mechanism here and it
    will automatically update in all commands.
    """
    click.echo(string)


def _should_configure_logger(logger: logging.Logger):
    """Determine whether a logger should be configured.

    Separated from configure_logger for ease of testing.
    """
    try:
        has_handlers = logger.hasHandlers()
    except AttributeError:
        # for LoggerAdapter classes
        has_handlers = logger.logger.hasHandlers()

    if has_handlers:
        return False

    # walk up the logging tree to see if any parent loggers are not default
    l = logger
    while (
        l.parent is not None  # not the root logger
        and l.level == logging.NOTSET  # level not already set
        and l.propagate  # configured to use parent when not set
    ):
        l = l.parent

    is_default = (l == logging.root and l.level == logging.WARNING)

    return is_default


def configure_logger(logger_name: str, level: int = logging.INFO, *,
                     handler: Optional[logging.Handler] = None):
    """Configure the logger at ``logger_name`` to be at ``level``.

    This is used to prevent accidentally overwriting existing logging
    configurations.

    This is particularly useful for setting INFO-level log messages to be
    seen in the CLI (with the default handler/formatter).

    Parameters
    ----------
    logger_name: str
        name of the logger to configure
    level: int
        level to set the logger to use, typically one of the constants
        defined in the ``logging`` module.
    """
    logger = logging.getLogger(logger_name)

    if _should_configure_logger(logger):
        logger.setLevel(level)
        if handler is not None:
            logger.addHandler(handler)


def print_duration(function: Callable) -> Callable:
    """
    Helper function to denote that a function should print a duration information.
    A function decorated with this decorator will print out the execution time of
    the decorated function.

    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()

        result = function(*args, **kwargs)

        end_time = datetime.now()
        duration = end_time - start_time
        write("\tDuration: " + str(duration) + "\n")
        return result

    return wrapper

