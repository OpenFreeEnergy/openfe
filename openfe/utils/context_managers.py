from contextlib import contextmanager

@contextmanager
def temp_env(environment_variables: dict[str, str]):
    """
    Temporarily update the global environment variables using the supplied values.

    Inspired by bespokefit
    <https://github.com/openforcefield/openff-bespokefit/blob/1bd79e9a9e4cea982153aed8e9cc6f8a37d65bd8/openff/bespokefit/utilities/_settings.py#L133>

    Parameters
    ----------
    environment_variables: dict[str, str]
        The variables which should be changed while the manager is active.
    """
    import os

    old_env = dict(os.environ)
    os.environ.update(environment_variables)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)
