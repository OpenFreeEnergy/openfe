from unittest.mock import patch, Mock

import pytest

import contextlib
from openfe.utils.system_probe import _get_gpu_info, _get_hostname, _probe_system


@contextlib.contextmanager
def patch_system():
    # single patch to fully patch the system
    patch_hostname = patch(
        "socket.gethostname",
        Mock(return_value="mock-hostname")
    )
    # assumes that each shell command is called in only one way
    cmd_to_output = {
        'nvidia-smi': (
            b"gpu_uuid,gpu_name,compute_mode\n"
            b"GPU-UUID-1,GeForce RTX 3080,Default\n"
            b"GPU-UUID-2,GeForce GTX 1660,Exclusive Process\n"
        ),
    }
    patch_check_output = patch(
        "subprocess.check_output",
        Mock(side_effect=lambda args, **kwargs: cmd_to_output[args[0]])
    )
    with contextlib.ExitStack() as stack:
        for ctx in [patch_hostname, patch_check_output]:
            stack.enter_context(ctx)

        yield stack


def test_get_hostname():
    with patch_system():
        hostname = _get_hostname()
        assert hostname == "mock-hostname"


def test_get_gpu_info():
    with patch_system():
        gpu_info = _get_gpu_info()
        expected_gpu_info = {
            "GPU-UUID-1": {"gpu_name": "GeForce RTX 3080", "compute_mode": "Default"},
            "GPU-UUID-2": {
                "gpu_name": "GeForce GTX 1660",
                "compute_mode": "Exclusive Process",
            },
        }
        assert gpu_info == expected_gpu_info


def test_probe_system():
    with patch_system():
        system_info = _probe_system()
        expected_system_info = {
            "system information": {
                "hostname": "mock-hostname",
                "gpu information": {
                    "GPU-UUID-1": {
                        "gpu_name": "GeForce RTX 3080",
                        "compute_mode": "Default",
                    },
                    "GPU-UUID-2": {
                        "gpu_name": "GeForce GTX 1660",
                        "compute_mode": "Exclusive Process",
                    },
                },
            }
        }
        assert system_info == expected_system_info
