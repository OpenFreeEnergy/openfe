import logging
import os
import socket
import subprocess

import psutil


def _get_disk_usage() -> dict[str, dict[str, str]]:

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

        # The maximum size of the process's virtual memory
        RLIMIT_AS = p.rlimit(psutil.RLIMIT_AS)
        mem = psutil.virtual_memory()

    info["RLIMIT_AS"] = RLIMIT_AS
    info["virtual_memory"] = mem

    return info


def _get_hostname() -> str:

    return socket.gethostname()


def _get_gpu_info() -> dict[str, dict[str, str]]:

    GPU_QUERY = "--query-gpu=gpu_uuid,gpu_name,compute_mode,pstate,temperature.gpu,utilization.memory,memory.total,driver_version,"

    try:
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", GPU_QUERY, "--format=csv"]
        ).decode("utf-8")
    except FileNotFoundError:
        logging.debug(
            "Error: nvidia-smi command not found. Make sure NVIDIA drivers are installed, this is expected if there is no GPU available"
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


def log_system_probe(level=logging.DEBUG):
    """Print the system information via configurable logging.

    This creates a logger tree under "{__name__}.log", allowing one to turn
    on/off logging of GPU info or hostname info the with
    "{__name__}.log.gpu" and "{__name__}.log.hostname" loggers.
    """
    basename = __name__ + ".log"
    base = logging.getLogger(basename)
    gpu = logging.getLogger(basename + ".gpu")
    hostname = logging.getLogger(basename + ".hostname")
    loggers = [base, gpu, hostname]
    if any(l.isEnabledFor(level) for l in loggers):
        sysinfo = _probe_system()['system information']
        base.log(level, "SYSTEM CONFIG DETAILS:")
        hostname.log(level, f"hostname: '{sysinfo['hostname']}'")
        if ...:
            for gpu_card in ...:
                gpu.log(level, f"GPU: {sysinfo['gpu information'][}")
            gpu.log(level, f"CUDA driver: {...}")
            gpu.log(level, f"CUDA toolkit: {...}")
        else:
            gpu.log(level, f"No CUDA found")

        base.log(level, f"Memory used: {...} ({...}%)")
        for disk in ...:
            base.log(level, f"Disk free: {...}")

if __name__ == "__main__":
    print(_probe_system())
