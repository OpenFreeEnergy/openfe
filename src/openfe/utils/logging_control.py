import logging
from abc import ABC, abstractmethod


class _BaseLogFilter(ABC):
    """Base class for log filters that handle string or list of strings.

    Parameters
    ----------
    strings : str or list of str
        String(s) to use in the filter logic
    """

    def __init__(self, strings: str | list[str]) -> None:
        if isinstance(strings, str):
            strings = [strings]
        self.strings: list[str] = strings

    @abstractmethod
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter method to be implemented by subclasses.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to filter/modify

        Returns
        -------
        bool
            True to allow the record, False to block it
        """
        ...


class _MsgIncludesStringFilter(_BaseLogFilter):
    """Logging filter to silence specific log messages.

    See https://docs.python.org/3/library/logging.html#filter-objects

    Parameters
    ----------
    strings : str or list of str
        If string(s) match in log messages (substring match) then the log record
        is suppressed
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records that contain any of the specified strings.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to filter

        Returns
        -------
        bool
            False if the record should be blocked, True if it should be logged
        """
        for string in self.strings:
            if string in record.msg:
                return False
        return True


class _AppendMsgFilter(_BaseLogFilter):
    """Logging filter to append a message to a specific log message.

    See https://docs.python.org/3/library/logging.html#filter-objects

    Parameters
    ----------
    strings : str or list of str
        Suffix text(s) to append to log messages
    """

    def __init__(self, strings: str | list[str]) -> None:
        super().__init__(strings)
        # Rename for clarity in this context
        self.suffixes = self.strings

    def filter(self, record: logging.LogRecord) -> bool:
        """Append suffix to log record message.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to modify

        Returns
        -------
        bool
            Always True to allow the record to be logged
        """
        for suffix in self.suffixes:
            # Only modify if not already appended (idempotent)
            if not record.msg.endswith(suffix):
                record.msg = f"{record.msg}{suffix}"
        return True


def _silence_message(msg: str | list[str], logger_names: str | list[str]) -> None:
    """Silence specific log messages from one or more loggers.

    Parameters
    ----------
    msg : str or list of str
        String(s) to match in log messages (substring match)
    logger_names : str or list of str
        Logger name(s) to apply the filter to

    Examples
    --------
    >>> _silence_message(
    ...     msg="****** PyMBAR will use 64-bit JAX! *******",
    ...     logger_names=["pymbar.timeseries", "pymbar.mbar_solvers"]
    ... )
    """
    if isinstance(logger_names, str):
        logger_names = [logger_names]

    filter_obj = _MsgIncludesStringFilter(msg)
    for name in logger_names:
        logging.getLogger(name).addFilter(filter_obj)


def _silence_logger(logger_names: str | list[str], level: int = logging.CRITICAL) -> None:
    """Completely silence one or more loggers.

    Parameters
    ----------
    logger_names : str or list of str
        Logger name(s) to silence
    level : int
        Set logger level (default: CRITICAL to silence everything)

    Examples
    --------
    >>> _silence_logger(logger_names=["urllib3", "requests"])
    """
    if isinstance(logger_names, str):
        logger_names = [logger_names]

    for name in logger_names:
        logging.getLogger(name).setLevel(level)


def _append_logger(suffix: str | list[str], logger_names: str | list[str]) -> None:
    """Append text to logger messages.

    Parameters
    ----------
    suffix : str or list of str
        Suffix text to append to log messages
    logger_names : str or list of str
        Logger name(s) to modify

    Examples
    --------
    >>> _append_logger(
    ...     suffix=" [DEPRECATED]",
    ...     logger_names="myapp"
    ... )
    """
    if isinstance(logger_names, str):
        logger_names = [logger_names]

    filter_obj = _AppendMsgFilter(suffix)
    for name in logger_names:
        logging.getLogger(name).addFilter(filter_obj)
