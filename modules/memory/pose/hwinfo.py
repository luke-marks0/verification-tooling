"""Collect hardware info for report headers."""

import os
import platform
import socket
import subprocess


def collect_hwinfo() -> dict:
    hostname = socket.gethostname()
    arch = platform.machine()

    # CPU model
    cpu_model = "unknown"
    try:
        r = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "Model name" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    # GPU name
    gpu_name = "unknown"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        gpu_name = r.stdout.strip()
    except Exception:
        pass

    # Instance type (Lambda-specific)
    instance_type = os.environ.get("LAMBDA_INSTANCE_TYPE", "unknown")

    return {
        "hostname": hostname,
        "arch": arch,
        "cpu_model": cpu_model,
        "gpu_name": gpu_name,
        "instance_type": instance_type,
    }
