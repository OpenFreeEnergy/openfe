import logging
import socket
import subprocess


def _get_hostname() -> str:
    """
    Get the hostname of the current machine.

    Returns
    -------
    str
        The hostname of the current machine.

    Notes
    -----
    This function uses the `socket` module to retrieve the standard hostname
    for the current machine. The hostname represents the name of the machine
    within a network environment.

    Examples
    --------
    >>> _get_hostname()
    'my-computer'
    """

    return socket.gethostname()


def _get_gpu_info() -> dict[str, dict[str, str]]:
    """
    Get GPU information using the `nvidia-smi` command.

    Returns
    -------
    Dict[str, Dict[str, str]]
        A dictionary containing GPU information. The keys of the outer dictionary
        are GPU UUIDs, and the values are dictionaries containing GPU information.
        The inner dictionaries have the following keys:
            - 'gpu_name': str
                The name of the GPU.
            - 'compute_mode': str
                The compute mode of the GPU.

    Raises
    ------
    FileNotFoundError
        If the `nvidia-smi` command is not found or NVIDIA drivers are not installed.
        An empty dictionary will be returned.

    Notes
    -----
    This function runs the `nvidia-smi` command to gather GPU information. It uses the `subprocess` module
    to execute the command and parses the CSV output to build the GPU information dictionary.
    If the `nvidia-smi` command is not found or no GPU is available, an empty dictionary is returned.

    The GPU information includes the GPU UUID, name, and compute mode.

    Examples
    --------
    >>> _get_gpu_info()
    {'GPU-UUID-1': {'gpu_name': 'GeForce RTX 3080', 'compute_mode': 'Default'},
     'GPU-UUID-2': {'gpu_name': 'GeForce GTX 1660', 'compute_mode': 'Exclusive Process'}}
    """

    GPU_QUERY = "--query-gpu=gpu_uuid,gpu_name,compute_mode"

    try:
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", GPU_QUERY, "--format=csv"]
        ).decode("utf-8")
    except FileNotFoundError:
        logging.debug(
            "Error: nvidia-smi command not found. Make sure NVIDIA drivers are installed, this is expected if there is no GPU available"
        )
        return {}

    nvidia_smi_output_lines = nvidia_smi_output.strip().split("\n")

    header = nvidia_smi_output_lines[0].split(",")

    # Parse each line as CSV and build the dictionary
    # Skip the header
    gpu_info: dict[str, dict] = {}
    for line in nvidia_smi_output_lines[1:]:
        data = line.split(",")
        # Get UUID of GPU
        gpu_uuid = data[0].strip()
        gpu_info[gpu_uuid] = {}

        # Stuff info we asked for into dict with UUID as key
        for i in range(1, len(header)):
            field_name = header[i].strip()
            gpu_info[gpu_uuid][field_name] = data[i].strip()

    return gpu_info


def _probe_system() -> dict:
    """
    Probe the system to gather hostname and GPU information.

    Returns
    -------
    dict
        A dictionary containing system information. The keys of the outer dictionary are:
            - 'system information': dict
                A nested dictionary containing the following system information:
                    - 'hostname': str
                        The hostname of the current machine.
                    - 'gpu information': dict
                        A dictionary containing GPU information. The keys of this dictionary
                        are GPU UUIDs, and the values are dictionaries containing GPU information.
                        The inner dictionaries have the following keys:
                            - 'gpu_name': str
                                The name of the GPU.
                            - 'compute_mode': str
                                The compute mode of the GPU.

    Notes
    -----
    This function calls the `_get_hostname()` and `_get_gpu_info()` functions to gather system information.
    The `_get_hostname()` function retrieves the standard hostname for the current machine using the `socket` module.
    The `_get_gpu_info()` function runs the `nvidia-smi` command (if available) to gather GPU information and returns a
    dictionary containing GPU UUIDs, names, and compute modes.

    If the `nvidia-smi` command is not found or no GPU is available, the 'gpu information' key in the returned dictionary
    will be an empty dictionary.

    Examples
    --------
    >>> _probe_system()
    {'system information': {'hostname': 'my-computer',
                            'gpu information': {'GPU-UUID-1': {'gpu_name': 'GeForce RTX 3080', 'compute_mode': 'Default'},
                                                'GPU-UUID-2': {'gpu_name': 'GeForce GTX 1660', 'compute_mode': 'Exclusive Process'}}}}
    """

    hostname = _get_hostname()
    gpu_info = _get_gpu_info()

    return {"system information": {"hostname": hostname, "gpu information": gpu_info}}
