"""Helper utilities for CLI tests"""

import click
import traceback


def assert_click_success(result: click.testing.Result): # -no-cov-
    """Pass through error message if a click test fails.
    Taken from https://github.com/openpathsampling/openpathsampling-cli/blob/main/paths_cli/commands/pathsampling.py
    """
    if result.exit_code != 0:
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    assert result.exit_code == 0
