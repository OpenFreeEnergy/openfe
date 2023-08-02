import logging
import os
import socket
import sys
import subprocess

import psutil


def _get_disk_usage() -> dict[str, dict[str, str]]:
    """
    Get disk usage information for all filesystems.

    Returns
    -------
    dict[str, dict[str, str]]
        A dictionary with filesystem names as keys and dictionaries containing disk
        usage information as values.

    Notes
    -----
    This function uses the 'df' command-line utility to gather disk usage information
    for all filesystems mounted on the system. The output is then processed to extract
    relevant information.

    The returned dictionary has filesystem names as keys, and each corresponding value
    is a dictionary containing the following disk usage information:
    - 'size': The total size of the filesystem.
    - 'used': The used space on the filesystem.
    - 'available': The available space on the filesystem.
    - 'percent_used': Percentage of the filesystem's space that is currently in use.
    - 'mount_point': The mount point directory where the filesystem is mounted.

    Note that the disk space values are represented as strings, which include units
    (e.g., 'G' for gigabytes, 'M' for megabytes). The function decodes the 'df'
    command's output using 'utf-8'.

    Raises
    ------
    subprocess.CalledProcessError
        If the 'df' command execution fails or returns a non-zero exit code.

    OSError
        If an operating system-related error occurs during the 'df' command execution.

    Returns an empty dictionary if the 'df' command output is empty or cannot be
    processed.

    Examples
    --------
    >>> _get_disk_usage()
    {
        '/dev/sda1': {
            'size': '30G',
            'used': '15G',
            'available': '14G',
            'percent_used': '52%',
            'mount_point': '/'
        },
        '/dev/sda2': {
            'size': '100G',
            'used': '20G',
            'available': '80G',
            'percent_used': '20%',
            'mount_point': '/home'
        },
        ...
    }
    """

    output = subprocess.check_output(["df", "-h"]).decode("utf-8")

    lines = output.strip().split(os.linesep)
    lines = lines[1:]
    disk_usage_dict = {}

    for line in lines:
        columns = line.split()

        filesystem = columns[0]
        size = columns[1]
        used = columns[2]
        available = columns[3]
        percent_used = columns[4]
        mount_point = columns[5]

        disk_usage_dict[filesystem] = {
            "size": size,
            "used": used,
            "available": available,
            "percent_used": percent_used,
            "mount_point": mount_point,
        }

    return disk_usage_dict


def _get_psutil_info() -> dict[str, dict[str, str]]:
    """
    Get process information using the psutil library.

    Returns
    -------
    dict[str, dict[str, str]]
        A dictionary containing various process information.

    Notes
    -----
    This function utilizes the psutil library to retrieve information about the current
    process and system memory.

    The returned dictionary includes the following process information:
    - 'cpu_percent': The percentage of CPU usage by the current process.
    - 'create_time': The timestamp indicating the process creation time.
    - 'exe': The absolute path to the executable file associated with the process.
    - 'memory_full_info': A dictionary containing detailed memory information about
                          the process.
    - 'memory_percent': The percentage of memory usage by the current process.
    - 'num_fds': The number of file descriptors used by the process.
    - 'pid': The Process ID (PID) of the current process.
    - 'status': The current status of the process
                (e.g., 'running', 'sleeping', 'stopped').

    Additionally, the dictionary includes the following system memory information:
    - 'RLIMIT_AS': The maximum size of the process's virtual memory.
    - 'virtual_memory': A dictionary containing various virtual memory statistics for
                        the system.

    Note that the memory-related values are represented as strings, which include units
    (e.g., 'MB', 'GB').

    Note that RLIMIT_AS key will be missing when this function is executed on macOS
    systems.

    Raises
    ------
    NoSuchProcess
        If the process with the specified PID (Process ID) does not exist or is not
        running.

    AccessDenied
        If access to the process information is denied due to permission restrictions.

    Examples
    --------
    >>> _get_psutil_info()
    {
        "memory_percent": 0.019870108903294176,
        "exe": "/usr/bin/python3.10",
        "pid": 1531909,
        "cpu_percent": 0.0,
        "create_time": 1690995569.42,
        "memory_full_info": {
            "rss": 13369344,
            "vms": 31834112,
            "shared": 6815744,
            "text": 2121728,
            "lib": 0,
            "data": 7827456,
            "dirty": 0,
            "uss": 10633216,
            "pss": 10646528,
            "swap": 0,
        },
        "status": "running",
        "num_fds": 4,
        "RLIMIT_AS": (-1, -1),
        "virtual_memory": {
            "total": 67283697664,
            "available": 32223358976,
            "percent": 52.1,
            "used": 29410000896,
            "free": 3407593472,
            "active": 33954336768,
            "inactive": 26209050624,
            "buffers": 144347136,
            "cached": 34321756160,
            "shared": 1021435904,
            "slab": 1520009216,
        },
    }
    """

    p = psutil.Process()

    with p.oneshot():
        info = p.as_dict(
            attrs=[
                "cpu_percent",
                "create_time",
                "exe",
                "memory_full_info",
                "memory_percent",
                "num_fds",
                "pid",
                "status",
            ]
        )
        # OSX doesn't have rlimit for Process
        if sys.platform != "darwin":
            RLIMIT_AS = p.rlimit(psutil.RLIMIT_AS)
            info["RLIMIT_AS"] = RLIMIT_AS

        # The maximum size of the process's virtual memory
        mem = psutil.virtual_memory()

    # memory_full_info is a named tuple, and we need to dict-ify it
    mem_full_info = info["memory_full_info"]._asdict()
    info["memory_full_info"] = mem_full_info

    info["virtual_memory"] = mem._asdict()

    return info


def _get_hostname() -> str:
    """
    Get the hostname of the current system.

    Returns
    -------
    str
        The hostname of the system.

    Notes
    -----
    This function uses the 'socket' library to retrieve the hostname of the current
    system.

    The returned hostname is a string representing the name of the system within a
    network.

    Raises
    ------
    socket.error
        If an error occurs while trying to fetch the hostname.

    Examples
    --------
    >>> _get_hostname()
    'winry-comp'
    """

    return socket.gethostname()


def _get_gpu_info() -> dict[str, dict[str, str]]:
    """
    Get GPU information using the 'nvidia-smi' command-line utility.

    Returns
    -------
    dict[str, dict[str, str]]
        A dictionary with GPU UUIDs as keys and dictionaries containing GPU information
        as values.

    Notes
    -----
    This function queries the NVIDIA System Management Interface ('nvidia-smi') to
    retrieve information about the available GPUs on the system.

    The returned dictionary includes the following GPU information for each
    detected GPU:
    - 'gpu_name': The name of the GPU.
    - 'compute_mode': The compute mode of the GPU.
    - 'pstate': The current performance state of the GPU.
    - 'temperature.gpu': The temperature of the GPU.
    - 'utilization.memory': The memory utilization of the GPU.
    - 'memory.total': The total memory available on the GPU.
    - 'driver_version': The version of the installed NVIDIA GPU driver.

    The GPU information is extracted from the output of the 'nvidia-smi' command, which
    is invoked with specific query parameters. The output is then parsed as CSV, and
    the relevant information is stored in the dictionary.

    Note that the GPU information values are represented as strings.
    Note that if no GPU is detected, an empty dictionary is returned.

    Examples
    --------
    >>> _get_gpu_info()
    {
        'GPU-UUID-1': {
            'name': 'NVIDIA GeForce RTX 3080',
            'compute_mode': 'Default',
            'pstate': 'P0',
            'temperature.gpu': '78 C',
            'utilization.memory [%]': '50 %',
            'memory.total [MiB]': '10.7 GB',
            'driver_version': '470.57.02',
        },
        'GPU-UUID-2': {
            'name': 'NVIDIA GeForce GTX 1660 Ti',
            'compute_mode': 'Default',
            'pstate': 'P2',
            'temperature.gpu': '65 C',
            'utilization.memory [%]': '30 %',
            'memory.total [MiB]': '5.8 GB',
            'driver_version': '470.57.02',
        },
        ...
    }
    """

    GPU_QUERY = (
        "--query-gpu=gpu_uuid,gpu_name,compute_mode,pstate,temperature.gpu,"
        "utilization.memory,memory.total,driver_version,"
    )

    try:
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", GPU_QUERY, "--format=csv"]
        ).decode("utf-8")
    except FileNotFoundError:
        logging.debug(
            "Error: nvidia-smi command not found. Make sure NVIDIA drivers are"
            " installed, this is expected if there is no GPU available"
        )
        return {}

    nvidia_smi_output_lines = nvidia_smi_output.strip().split(os.linesep)

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
    Probe the system and gather various system information.

    Returns
    -------
    dict
        A dictionary containing system information.

    Notes
    -----
    This function gathers information about the system by calling several internal
    functions.

    The returned dictionary contains the following system information:
    - 'system information': A dictionary containing various system-related details.
        - 'hostname': The hostname of the current system.
        - 'gpu information': GPU information retrieved using the '_get_gpu_info'
          function.
        - 'psutil information': Process and memory-related information obtained using
          the '_get_psutil_info' function.
        - 'disk usage information': Disk usage details for all filesystems, obtained
          through the '_get_disk_usage' function.

    Each nested dictionary provides specific details about the corresponding system
    component.

    Examples
    --------
    >>> _probe_system()
    {
        "system information": {
            "hostname": "winry-comp",
            "gpu information": {
                "GPU-5b97c87b-4646-cfdd-efd6-3ee9bb3b371d": {
                    "name": "NVIDIA GeForce RTX 2060",
                    "compute_mode": "Default",
                    "pstate": "P0",
                    "temperature.gpu": "48",
                    "utilization.memory [%]": "0 %",
                    "memory.total [MiB]": "6144 MiB",
                    "driver_version": "525.116.04",
                }
            },
            "psutil information": {
                "exe": "/home/winry/micromamba/envs/openfe/bin/python3.10",
                "memory_percent": 0.02006491389254216,
                "create_time": 1690996699.21,
                "status": "running",
                "pid": 1549447,
                "num_fds": 4,
                "memory_full_info": {
                    "rss": 13500416,
                    "vms": 31850496,
                    "shared": 6946816,
                    "text": 2121728,
                    "lib": 0,
                    "data": 7843840,
                    "dirty": 0,
                    "uss": 10752000,
                    "pss": 10765312,
                    "swap": 0,
                },
                "cpu_percent": 0.0,
                "RLIMIT_AS": (-1, -1),
                "virtual_memory": {
                    "total": 67283697664,
                    "available": 31865221120,
                    "percent": 52.6,
                    "used": 29719117824,
                    "free": 2608443392,
                    "active": 34446774272,
                    "inactive": 26320441344,
                    "buffers": 168124416,
                    "cached": 34788012032,
                    "shared": 1069752320,
                    "slab": 1520705536,
                },
            },
            "disk usage information": {
                "/dev/mapper/data-root": {
                    "size": "1.8T",
                    "used": "626G",
                    "available": "1.1T",
                    "percent_used": "37%",
                    "mount_point": "/",
                },
                "/dev/dm-3": {
                    "size": "3.7T",
                    "used": "1.6T",
                    "available": "2.2T",
                    "percent_used": "42%",
                    "mount_point": "/mnt/data",
                },
            },
        }
    }
    """

    hostname = _get_hostname()
    gpu_info = _get_gpu_info()
    psutil_info = _get_psutil_info()
    disk_usage_info = _get_disk_usage()

    return {
        "system information": {
            "hostname": hostname,
            "gpu information": gpu_info,
            "psutil information": psutil_info,
            "disk usage information": disk_usage_info,
        }
    }


if __name__ == "__main__":
    print(_probe_system())
