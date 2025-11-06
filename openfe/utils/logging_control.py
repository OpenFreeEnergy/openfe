import logging

class MsgIncludesStringFilter:
    """Logging filter to silence specfic log messages.

    See https://docs.python.org/3/library/logging.html#filter-objects

    Parameters
    ----------
    strings : str or list of str
        If string(s) match in log messages (substring match) then the log record
        is suppressed
    """

    def __init__(self, strings: str | list[str]) -> None:
        if isinstance(strings, str):
            strings = [strings]
        self.strings = strings

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
            return not string in record.msg


class AppendMsgFilter:
    """Logging filter to append a message to a specfic log message.

    See https://docs.python.org/3/library/logging.html#filter-objects

    """
    def __init__(self, suffix: str | list[str]) -> None:
        if isinstance(suffix, str):
            suffix = [suffix]
        self.suffixes = suffix

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


class LogControl:
    """Easy-to-use logging control for third-party packages."""

    def __init__(self):
        self._filters = []

    @staticmethod
    def silence_message(msg, logger_names):
        """Silence specific log messages from one or more loggers.

        Parameters
        ----------
        msg : str or list of str
            String(s) to match in log messages (substring match)
        logger_names : str or list of str
            Logger name(s) to apply the filter to

        Examples
        --------
        >>> LogControl.silence_message(
        ...     msg="****** PyMBAR will use 64-bit JAX! *******",
        ...     logger_names=["pymbar.timeseries", "pymbar.mbar_solvers"]
        ... )
        >>> LogControl.silence_message(
        ...     msg=["warning 1", "warning 2", "warning 3"],
        ...     logger_names="some.package"
        ... )
        """
        # Handle single string or list for both parameters
        if isinstance(logger_names, str):
            logger_names = [logger_names]

        if isinstance(msg, str):
            msg = [msg]

        # Create a filter for each message
        for message in msg:
            filter_obj = MsgIncludesStringFilter(message)
            for name in logger_names:
                logging.getLogger(name).addFilter(filter_obj)

    @staticmethod
    def silence_logger(logger_names, level=logging.CRITICAL):
        """Completely silence one or more loggers.

        Parameters
        ----------
        logger_names : str or list of str
            Logger name(s) to silence
        level : int
            Set logger level (default: CRITICAL to silence everything)

        Examples
        --------
        >>> LogControl.silence_logger(logger_names=["urllib3", "requests"])
        >>> LogControl.silence_logger(logger_names="noisy.package")
        """
        if isinstance(logger_names, str):
            logger_names = [logger_names]

        for name in logger_names:
            logging.getLogger(name).setLevel(level)


    @staticmethod
    def append_logger(suffix: str | list[str], logger_names: str | list[str]):
        """Add extra context to logger messages.

        Parameters
        ----------
        logger_names : str or list of str
            Logger name(s) to enhance
        formatter : callable, optional
            Function that takes a LogRecord and returns modified msg
        extra_fields : dict, optional
            Extra fields to add to LogRecord

        Examples
        --------
        >>> LogControl.append_logger(
        ...     logger_names="myapp",
        ...     extra_fields={'version': '1.0', 'env': 'prod'}
        ... )
        """
        if isinstance(logger_names, str):
            logger_names = [logger_names]

        filter_obj = AppendMsgFilter(suffix)
        for name in logger_names:
            logging.getLogger(name).addFilter(filter_obj)
