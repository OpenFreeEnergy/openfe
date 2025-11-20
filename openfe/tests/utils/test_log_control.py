import logging

import pytest

from openfe.utils import logging_control
from openfe.utils.logging_control import (
    _AppendMsgFilter,
    _BaseLogFilter,
    _MsgIncludesStringFilter,
)


@pytest.fixture
def logger():
    """Create a test logger with a handler that captures log records."""
    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.DEBUG)
    test_logger.handlers = []  # Clear any existing handlers

    # Create a handler that stores log records
    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = ListHandler()
    test_logger.addHandler(handler)

    yield test_logger, handler

    # Cleanup
    test_logger.handlers = []
    test_logger.filters = []


class Test_MsgIncludesStringFilter:
    """Tests for _MsgIncludesStringFilter."""

    def test_single_string_blocks_matching_message(self):
        """Test that a single string blocks messages containing it."""
        filter_obj = _MsgIncludesStringFilter("block this")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Please block this message",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False

    def test_single_string_allows_non_matching_message(self):
        """Test that messages not containing the string are allowed."""
        filter_obj = _MsgIncludesStringFilter("block this")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="This is fine",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True

    def test_list_of_strings_blocks_any_match(self):
        """Test that any string in the list blocks the message."""
        filter_obj = _MsgIncludesStringFilter(["warning1", "warning2", "warning3"])

        record1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="This has warning1 in it",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record1) is False

        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="This has warning3 in it",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record2) is False

    def test_list_allows_non_matching_messages(self):
        """Test that messages not matching any string are allowed."""
        filter_obj = _MsgIncludesStringFilter(["warning1", "warning2"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="This is completely different",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True

    def test_substring_matching(self):
        """Test that substring matching works correctly."""
        filter_obj = _MsgIncludesStringFilter("JAX")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="****** PyMBAR will use 64-bit JAX! *******",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False

    def test_case_sensitive_matching(self):
        """Test that matching is case-sensitive."""
        filter_obj = _MsgIncludesStringFilter("Error")

        record1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="This has Error in it",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record1) is False

        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="This has error in it",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record2) is True


class Test_AppendMsgFilter:
    """Tests for _AppendMsgFilter."""

    def test_single_suffix_appends(self):
        """Test that a single suffix is appended to the message."""
        filter_obj = _AppendMsgFilter(" [DEPRECATED]")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Original message",
            args=(),
            exc_info=None,
        )
        result = filter_obj.filter(record)
        assert result is True
        assert record.msg == "Original message [DEPRECATED]"

    def test_multiple_suffixes_append_in_order(self):
        """Test that multiple suffixes are appended in order."""
        filter_obj = _AppendMsgFilter([" [DEPRECATED]", " - see docs"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Original message",
            args=(),
            exc_info=None,
        )
        filter_obj.filter(record)
        assert record.msg == "Original message [DEPRECATED] - see docs"

    def test_idempotent_single_suffix(self):
        """Test that applying the same suffix twice is idempotent."""
        filter_obj = _AppendMsgFilter(" [DEPRECATED]")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Original message [DEPRECATED]",
            args=(),
            exc_info=None,
        )
        filter_obj.filter(record)
        assert record.msg == "Original message [DEPRECATED]"

    def test_idempotent_multiple_suffixes(self):
        """Test idempotency with multiple suffixes."""
        filter_obj = _AppendMsgFilter([" [DEPRECATED] - see docs", " [DEPRECATED] - see docs"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Original message [DEPRECATED] - see docs",
            args=(),
            exc_info=None,
        )
        filter_obj.filter(record)
        assert record.msg == "Original message [DEPRECATED] - see docs"

    def test_always_returns_true(self):
        """Test that the filter always returns True to allow logging."""
        filter_obj = _AppendMsgFilter(" [INFO]")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Message",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True


class Testlogging_control:
    """Tests for logging_control module."""

    def test__silence_message_single_string_single_logger(self, logger):
        """Test silencing a single message from a single logger."""
        test_logger, handler = logger

        logging_control._silence_message(msg="block this", logger_names="test_logger")

        test_logger.info("block this message")
        test_logger.info("allow this message")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "allow this message"

    def test__silence_message_multiple_strings_single_logger(self, logger):
        """Test silencing multiple messages from a single logger."""
        test_logger, handler = logger

        logging_control._silence_message(msg=["warning1", "warning2"], logger_names="test_logger")

        test_logger.info("This has warning1")
        test_logger.info("This has warning2")
        test_logger.info("This is fine")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "This is fine"

    def test__silence_message_single_string_multiple_loggers(self):
        """Test silencing a message from multiple loggers."""
        logger1 = logging.getLogger("test_logger1")
        logger2 = logging.getLogger("test_logger2")

        # Clear any existing filters
        logger1.filters = []
        logger2.filters = []

        logging_control._silence_message(
            msg="block this", logger_names=["test_logger1", "test_logger2"]
        )

        assert len(logger1.filters) == 1
        assert len(logger2.filters) == 1

        # Cleanup
        logger1.filters = []
        logger2.filters = []

    def test__silence_message_multiple_strings_multiple_loggers(self):
        """Test silencing multiple messages from multiple loggers."""
        logger1 = logging.getLogger("test_logger1")
        logger2 = logging.getLogger("test_logger2")

        logger1.filters = []
        logger2.filters = []

        logging_control._silence_message(
            msg=["warning1", "warning2"], logger_names=["test_logger1", "test_logger2"]
        )

        # Should have one filter per logger (not one per message)
        assert len(logger1.filters) == 1
        assert len(logger2.filters) == 1

        # Cleanup
        logger1.filters = []
        logger2.filters = []

    def test__silence_logger_single(self, logger):
        """Test completely silencing a single logger."""
        test_logger, handler = logger

        logging_control._silence_logger(logger_names="test_logger")

        test_logger.debug("debug message")
        test_logger.info("info message")
        test_logger.warning("warning message")
        test_logger.error("error message")

        # All messages should be blocked
        assert len(handler.records) == 0

    def test__silence_logger_multiple(self):
        """Test silencing multiple loggers."""
        logger1 = logging.getLogger("test_logger1")
        logger2 = logging.getLogger("test_logger2")

        original_level1 = logger1.level
        original_level2 = logger2.level

        logging_control._silence_logger(logger_names=["test_logger1", "test_logger2"])

        assert logger1.level == logging.CRITICAL
        assert logger2.level == logging.CRITICAL

        # Cleanup
        logger1.setLevel(original_level1)
        logger2.setLevel(original_level2)

    def test__silence_logger_custom_level(self, logger):
        """Test silencing logger with custom level."""
        test_logger, handler = logger

        logging_control._silence_logger(logger_names="test_logger", level=logging.ERROR)

        test_logger.debug("debug message")
        test_logger.info("info message")
        test_logger.warning("warning message")
        test_logger.error("error message")

        # Only error and above should pass through
        assert len(handler.records) == 1
        assert handler.records[0].msg == "error message"

    def test__append_logger_single_suffix_single_logger(self, logger):
        """Test appending a single suffix to a single logger."""
        test_logger, handler = logger

        logging_control._append_logger(suffix=" [DEPRECATED]", logger_names="test_logger")

        test_logger.info("Original message")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "Original message [DEPRECATED]"

    def test__append_logger_multiple_suffixes(self, logger):
        """Test appending multiple suffixes."""
        test_logger, handler = logger

        logging_control._append_logger(
            suffix=[" [DEPRECATED]", " - see docs"], logger_names="test_logger"
        )

        test_logger.info("Original message")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "Original message [DEPRECATED] - see docs"

    def test__append_logger_multiple_loggers(self):
        """Test appending to multiple loggers."""
        logger1 = logging.getLogger("test_logger1")
        logger2 = logging.getLogger("test_logger2")

        logger1.filters = []
        logger2.filters = []

        logging_control._append_logger(
            suffix=" [INFO]", logger_names=["test_logger1", "test_logger2"]
        )

        assert len(logger1.filters) == 1
        assert len(logger2.filters) == 1

        # Cleanup
        logger1.filters = []
        logger2.filters = []

    def test_pymbar_example(self, logger):
        """Test the PyMBAR use case."""
        test_logger, handler = logger

        logging_control._silence_message(
            msg="****** PyMBAR will use 64-bit JAX! *******", logger_names="test_logger"
        )

        test_logger.info("****** PyMBAR will use 64-bit JAX! *******")
        test_logger.info("Other message")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "Other message"

    def test_combining_multiple_controls(self, logger):
        """Test combining silence and append on the same logger."""
        test_logger, handler = logger

        logging_control._silence_message(msg="block", logger_names="test_logger")
        logging_control._append_logger(suffix=" [INFO]", logger_names="test_logger")

        test_logger.info("block this")
        test_logger.info("allow this")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "allow this [INFO]"


class TestBaseLogFilter:
    """Tests for _BaseLogFilter base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that _BaseLogFilter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            _BaseLogFilter("test")

    def test_subclass_must_implement_filter(self):
        """Test that subclasses must implement the filter method."""

        class IncompleteFilter(_BaseLogFilter):
            pass

        with pytest.raises(TypeError):
            IncompleteFilter("test")

    def test_subclass_with_filter_works(self):
        """Test that a proper subclass can be instantiated."""

        class CompleteFilter(_BaseLogFilter):
            def filter(self, record):
                return True

        filter_obj = CompleteFilter("test")
        assert filter_obj.strings == ["test"]

    def test_string_conversion_to_list(self):
        """Test that single strings are converted to lists."""
        filter_obj = _MsgIncludesStringFilter("single")
        assert isinstance(filter_obj.strings, list)
        assert filter_obj.strings == ["single"]

    def test_list_stays_as_list(self):
        """Test that lists remain as lists."""
        filter_obj = _MsgIncludesStringFilter(["one", "two", "three"])
        assert isinstance(filter_obj.strings, list)
        assert filter_obj.strings == ["one", "two", "three"]


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_string(self, logger):
        """Test behavior with empty string."""
        test_logger, handler = logger

        logging_control._silence_message(msg="", logger_names="test_logger")

        test_logger.info("message")

        # Empty string matches everything as substring
        assert len(handler.records) == 0

    def test_empty_list(self, logger):
        """Test behavior with empty list."""
        test_logger, handler = logger

        logging_control._silence_message(msg=[], logger_names="test_logger")

        test_logger.info("message")

        # Empty list should not block anything
        assert len(handler.records) == 1

    def test_special_characters_in_message(self, logger):
        """Test that special characters are handled correctly."""
        test_logger, handler = logger

        logging_control._silence_message(
            msg="[WARNING] *special* $chars$", logger_names="test_logger"
        )

        test_logger.info("[WARNING] *special* $chars$ in message")
        test_logger.info("normal message")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "normal message"

    def test_unicode_characters(self, logger):
        """Test that unicode characters work correctly."""
        test_logger, handler = logger

        logging_control._silence_message(msg="🚫 blocked", logger_names="test_logger")
        logging_control._append_logger(suffix=" ✅", logger_names="test_logger")

        test_logger.info("🚫 blocked message")
        test_logger.info("allowed message")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "allowed message ✅"

    def test_very_long_message(self, logger):
        """Test handling of very long messages."""
        test_logger, handler = logger

        long_msg = "x" * 10000
        logging_control._silence_message(msg="needle", logger_names="test_logger")

        test_logger.info(long_msg + "needle" + long_msg)
        test_logger.info("short message")

        assert len(handler.records) == 1
        assert handler.records[0].msg == "short message"
