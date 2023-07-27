import pytest
from unittest.mock import patch
from openfe.utils.system_probe import _get_hostname, _get_gpu_info, _probe_system

# Mocking the socket.gethostname() function
def mock_gethostname():
    return "mock-hostname"

# Mocking the subprocess.check_output() function
def mock_check_output(command, *args, **kwargs):
    return b"gpu_uuid,gpu_name,compute_mode\nGPU-UUID-1,GeForce RTX 3080,Default\nGPU-UUID-2,GeForce GTX 1660,Exclusive Process\n"

def test_get_hostname():
    with patch("socket.gethostname", mock_gethostname):
        hostname = _get_hostname()
        assert hostname == "mock-hostname"

def test_get_gpu_info():
    with patch("subprocess.check_output", mock_check_output):
        gpu_info = _get_gpu_info()
        expected_gpu_info = {
            "GPU-UUID-1": {"gpu_name": "GeForce RTX 3080", "compute_mode": "Default"},
            "GPU-UUID-2": {"gpu_name": "GeForce GTX 1660", "compute_mode": "Exclusive Process"},
        }
        assert gpu_info == expected_gpu_info

def test_probe_system():
    with patch("socket.gethostname", mock_gethostname):
        with patch("subprocess.check_output", mock_check_output):
            system_info = _probe_system()
            expected_system_info = {
                "system information": {
                    "hostname": "mock-hostname",
                    "gpu information": {
                        "GPU-UUID-1": {"gpu_name": "GeForce RTX 3080", "compute_mode": "Default"},
                        "GPU-UUID-2": {"gpu_name": "GeForce GTX 1660", "compute_mode": "Exclusive Process"},
                    },
                }
            }
            assert system_info == expected_system_info
