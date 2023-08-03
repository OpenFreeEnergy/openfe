import contextlib
from collections import namedtuple
import logging
import sys
from unittest.mock import Mock, patch

import psutil
import pytest

from openfe.utils.system_probe import (
    _get_disk_usage,
    _get_gpu_info,
    _get_hostname,
    _get_psutil_info,
    _probe_system,
    log_system_probe,
)


# Named tuples from https://github.com/giampaolo/psutil/blob/master/psutil/_pslinux.py
svmem = namedtuple(
    "svmem",
    [
        "total",
        "available",
        "percent",
        "used",
        "free",
        "active",
        "inactive",
        "buffers",
        "cached",
        "shared",
        "slab",
    ],
)
pfullmem = namedtuple(
    "pfullmem",
    ["rss", "vms", "shared", "text", "lib", "data", "dirty", "uss", "pss", "swap"],
)


EXPECTED_SYSTEM_INFO = {
    "system information": {
        "hostname": "mock-hostname",
        "gpu information": {
            "GPU-UUID-1": {
                "name": "NVIDIA GeForce RTX 2060",
                "compute_mode": "Default",
                "pstate": "P8",
                "temperature.gpu": "47",
                "utilization.memory [%]": "6 %",
                "memory.total [MiB]": "6144 MiB",
                "driver_version": "525.116.04",
            },
            "GPU-UUID-2": {
                "name": "NVIDIA GeForce RTX 2060",
                "compute_mode": "Default",
                "pstate": "P8",
                "temperature.gpu": "47",
                "utilization.memory [%]": "6 %",
                "memory.total [MiB]": "6144 MiB",
                "driver_version": "525.116.04",
            },
        },
        "psutil information": {
            "pid": 1590579,
            "status": "running",
            "exe": "/home/winry/micromamba/envs/openfe/bin/python3.10",
            "cpu_percent": 0.0,
            "num_fds": 4,
            "create_time": 1690999298.62,
            "memory_percent": 0.02006491389254216,
            "memory_full_info": {
                "rss": 13500416,
                "vms": 31858688,
                "shared": 6946816,
                "text": 2121728,
                "lib": 0,
                "data": 7852032,
                "dirty": 0,
                "uss": 10764288,
                "pss": 10777600,
                "swap": 0,
            },
            "RLIMIT_AS": (-1, -1),
            "virtual_memory": {
                "total": 67283697664,
                "available": 31731806208,
                "percent": 52.8,
                "used": 29899350016,
                "free": 3136847872,
                "active": 25971789824,
                "inactive": 34514595840,
                "buffers": 136404992,
                "cached": 34111094784,
                "shared": 1021571072,
                "slab": 1518297088,
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


@contextlib.contextmanager
def patch_system():
    # single patch to fully patch the system
    patch_hostname = patch("socket.gethostname", Mock(return_value="mock-hostname"))

    patch_psutil_Process_as_dict = patch(
        "psutil.Process.as_dict",
        Mock(
            return_value={
                "pid": 1590579,
                "status": "running",
                "exe": "/home/winry/micromamba/envs/openfe/bin/python3.10",
                "cpu_percent": 0.0,
                "num_fds": 4,
                "create_time": 1690999298.62,
                "memory_percent": 0.02006491389254216,
                "memory_full_info": pfullmem(
                    rss=13500416,
                    vms=31858688,
                    shared=6946816,
                    text=2121728,
                    lib=0,
                    data=7852032,
                    dirty=0,
                    uss=10764288,
                    pss=10777600,
                    swap=0,
                ),
            }
        ),
    )
    # Since this attribute doesn't exist on OSX, we have to create it
    patch_psutil_Process_rlimit = patch(
        "psutil.Process.rlimit", Mock(return_value=(-1, -1))
    )
    patch_psutil_virtual_memory = patch(
        "psutil.virtual_memory",
        Mock(
            return_value=svmem(
                total=67283697664,
                available=31731806208,
                percent=52.8,
                used=29899350016,
                free=3136847872,
                active=25971789824,
                inactive=34514595840,
                buffers=136404992,
                cached=34111094784,
                shared=1021571072,
                slab=1518297088,
            )
        ),
    )

    # assumes that each shell command is called in only one way
    cmd_to_output = {
        "nvidia-smi": (
            b"uuid, name, compute_mode, pstate, temperature.gpu, utilization.memory [%], memory.total [MiB], driver_version\n"
            b"GPU-UUID-1, NVIDIA GeForce RTX 2060, Default, P8, 47, 6 %, 6144 MiB, 525.116.04\n"
            b"GPU-UUID-2, NVIDIA GeForce RTX 2060, Default, P8, 47, 6 %, 6144 MiB, 525.116.04\n"
        ),
        "df": (
            b"Filesystem             Size  Used Avail Use% Mounted on\n"
            b"/dev/mapper/data-root  1.8T  626G  1.1T  37% /\n"
            b"/dev/dm-3              3.7T  1.6T  2.2T  42% /mnt/data\n"
        ),
    }
    patch_check_output = patch(
        "subprocess.check_output",
        Mock(side_effect=lambda args, **kwargs: cmd_to_output[args[0]]),
    )
    with contextlib.ExitStack() as stack:
        for ctx in [
            patch_hostname,
            patch_psutil_Process_as_dict,
            patch_check_output,
            patch_psutil_Process_rlimit,
            patch_psutil_virtual_memory,
        ]:
            stack.enter_context(ctx)

        yield stack


@pytest.mark.skipif(
    sys.platform == "darwin", reason="test requires psutil.Process.rlimit"
)
def test_get_hostname():
    with patch_system():
        hostname = _get_hostname()
        assert hostname == "mock-hostname"


@pytest.mark.skipif(
    sys.platform == "darwin", reason="test requires psutil.Process.rlimit"
)
def test_get_gpu_info():
    with patch_system():
        gpu_info = _get_gpu_info()
        expected_gpu_info = {
            "GPU-UUID-1": {
                "name": "NVIDIA GeForce RTX 2060",
                "compute_mode": "Default",
                "pstate": "P8",
                "temperature.gpu": "47",
                "utilization.memory [%]": "6 %",
                "memory.total [MiB]": "6144 MiB",
                "driver_version": "525.116.04",
            },
            "GPU-UUID-2": {
                "name": "NVIDIA GeForce RTX 2060",
                "compute_mode": "Default",
                "pstate": "P8",
                "temperature.gpu": "47",
                "utilization.memory [%]": "6 %",
                "memory.total [MiB]": "6144 MiB",
                "driver_version": "525.116.04",
            },
        }
        assert gpu_info == expected_gpu_info


@pytest.mark.skipif(
    sys.platform == "darwin", reason="test requires psutil.Process.rlimit"
)
def test_get_psutil_info():
    with patch_system():
        psutil_info = _get_psutil_info()
        expected_psutil_info = {
            "pid": 1590579,
            "status": "running",
            "exe": "/home/winry/micromamba/envs/openfe/bin/python3.10",
            "cpu_percent": 0.0,
            "num_fds": 4,
            "create_time": 1690999298.62,
            "memory_percent": 0.02006491389254216,
            "memory_full_info": {
                "rss": 13500416,
                "vms": 31858688,
                "shared": 6946816,
                "text": 2121728,
                "lib": 0,
                "data": 7852032,
                "dirty": 0,
                "uss": 10764288,
                "pss": 10777600,
                "swap": 0,
            },
            "RLIMIT_AS": (-1, -1),
            "virtual_memory": {
                "total": 67283697664,
                "available": 31731806208,
                "percent": 52.8,
                "used": 29899350016,
                "free": 3136847872,
                "active": 25971789824,
                "inactive": 34514595840,
                "buffers": 136404992,
                "cached": 34111094784,
                "shared": 1021571072,
                "slab": 1518297088,
            },
        }
        assert psutil_info == expected_psutil_info


@pytest.mark.skipif(
    sys.platform == "darwin", reason="test requires psutil.Process.rlimit"
)
def test_get_disk_usage():
    with patch_system():
        disk_info = _get_disk_usage()
        expected_disk_info = {
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
        }
        assert disk_info == expected_disk_info


@pytest.mark.skipif(
    sys.platform == "darwin", reason="test requires psutil.Process.rlimit"
)
def test_probe_system():
    with patch_system():
        system_info = _probe_system()
        expected_system_info = {
            "system information": {
                "hostname": "mock-hostname",
                "gpu information": {
                    "GPU-UUID-1": {
                        "name": "NVIDIA GeForce RTX 2060",
                        "compute_mode": "Default",
                        "pstate": "P8",
                        "temperature.gpu": "47",
                        "utilization.memory [%]": "6 %",
                        "memory.total [MiB]": "6144 MiB",
                        "driver_version": "525.116.04",
                    },
                    "GPU-UUID-2": {
                        "name": "NVIDIA GeForce RTX 2060",
                        "compute_mode": "Default",
                        "pstate": "P8",
                        "temperature.gpu": "47",
                        "utilization.memory [%]": "6 %",
                        "memory.total [MiB]": "6144 MiB",
                        "driver_version": "525.116.04",
                    },
                },
                "psutil information": {
                    "pid": 1590579,
                    "status": "running",
                    "exe": "/home/winry/micromamba/envs/openfe/bin/python3.10",
                    "cpu_percent": 0.0,
                    "num_fds": 4,
                    "create_time": 1690999298.62,
                    "memory_percent": 0.02006491389254216,
                    "memory_full_info": {
                        "rss": 13500416,
                        "vms": 31858688,
                        "shared": 6946816,
                        "text": 2121728,
                        "lib": 0,
                        "data": 7852032,
                        "dirty": 0,
                        "uss": 10764288,
                        "pss": 10777600,
                        "swap": 0,
                    },
                    "RLIMIT_AS": (-1, -1),
                    "virtual_memory": {
                        "total": 67283697664,
                        "available": 31731806208,
                        "percent": 52.8,
                        "used": 29899350016,
                        "free": 3136847872,
                        "active": 25971789824,
                        "inactive": 34514595840,
                        "buffers": 136404992,
                        "cached": 34111094784,
                        "shared": 1021571072,
                        "slab": 1518297088,
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

        assert system_info == expected_system_info


def test_probe_system_smoke_test():
    _probe_system()


def test_log_system_probe_unconfigured():
    # if probe loggers aren't configured to run, then we shouldn't even call
    # _probe_system()
    logger_names = [
        'openfe.utils.system_probe.log',
        'openfe.utils.system_probe.log.gpu',
        'openfe.utils.system_probe.log.hostname',
    ]
    # check that initial conditions are as expected
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        assert not logger.isEnabledFor(logging.DEBUG)

    sysprobe_mock = Mock(return_value=EXPECTED_SYSTEM_INFO)
    with patch('openfe.utils.system_probe._probe_system', sysprobe_mock):
        log_system_probe(logging.DEBUG)
        assert sysprobe_mock.call_count == 0

    # now check that it does get called if we use a level that will emit
    # (this is effectively tests that the previous assert isn't a false
    # positive)
    with patch('openfe.utils.system_probe._probe_system', sysprobe_mock):
        log_system_probe(logging.WARNING)
        assert sysprobe_mock.call_count == 1


def test_log_system_probe(caplog):
    # this checks that the expected contents show up in log_system_probe
    sysprobe_mock = Mock(return_value=EXPECTED_SYSTEM_INFO)
    with patch('openfe.utils.system_probe._probe_system', sysprobe_mock):
        with caplog.at_level(logging.DEBUG):
            log_system_probe()

    expected = [
        "hostname: 'mock-hostname'",
        "GPU: uuid='GPU-UUID-1' NVIDIA GeForce RTX 2060 mode=Default",
        "GPU: uuid='GPU-UUID-2' NVIDIA GeForce RTX 2060 mode=Default",
        "Memory used: 27.8G (52.8%)",
        "/dev/mapper/data-root: 37% full (1.1T free)",
        "/dev/dm-3: 42% full (2.2T free)"
    ]
    for line in expected:
        assert line in caplog.text
