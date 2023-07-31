import os
import pytest

from unittest.mock import patch
import logging
import contextlib

from openfecli.utils import (
    import_thing, _should_configure_logger, configure_logger
)

# looks like this can't be done as a fixture; related to
# https://github.com/pytest-dev/pytest/issues/2974
@contextlib.contextmanager
def patch_root_logger():
    # use this to hide away some handlers that pytest attaches
    old_manager = logging.Logger.manager
    old_root = logging.root
    new_root = logging.RootLogger(logging.WARNING)
    manager = logging.Manager(new_root)

    logging.Logger.manager = manager
    logging.root = new_root
    yield new_root
    logging.Logger.manager = old_manager
    logging.root = old_root


@pytest.mark.parametrize('import_string,expected', [
    ('os.path.exists', os.path.exists),
    ('os.getcwd', os.getcwd),
    ('os', os),
])
def test_import_thing(import_string, expected):
    assert import_thing(import_string) is expected


def test_import_thing_import_error():
    with pytest.raises(ImportError):
        import_thing('foo.bar')


def test_import_thing_attribute_error():
    with pytest.raises(AttributeError):
        import_thing('os.foo')


@pytest.mark.parametrize("logger_name, expected", [
    ("default", True), ("default.default", True),
    ("level", False), ("level.default", False),
    ("handler", False), ("handler.default", False),
    ("default.noprop", False)
])
def test_should_configure_logger(logger_name, expected):
    with patch_root_logger():
        logging.getLogger("level").setLevel(logging.INFO)
        logging.getLogger("handler").addHandler(logging.NullHandler())
        logging.getLogger("default.noprop").propagate = False
        logger = logging.getLogger(logger_name)
        assert _should_configure_logger(logger) == expected

def test_root_logger_level_configured():
    with patch_root_logger():
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        logger = logging.getLogger("default.default")
        assert not _should_configure_logger(logger)


@pytest.mark.parametrize('with_handler', [True, False])
def test_configure_logger(with_handler):
    handler = logging.NullHandler() if with_handler else None
    expected_handlers = [handler] if handler else []
    with patch_root_logger():
        configure_logger('default.default', handler=handler)
        logger = logging.getLogger('default.default')
        parent = logging.getLogger('default')
        assert logger.isEnabledFor(logging.INFO)
        assert not parent.isEnabledFor(logging.INFO)
        assert logger.handlers == expected_handlers
        assert parent.handlers == []
