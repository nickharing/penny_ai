#!/usr/bin/env python3
"""
System and Environment Info Script
----------------------------------
Lists hardware capabilities, installed Python libraries, and other useful general information.
"""

import platform
import sys
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pkg_resources
except ImportError:
    pkg_resources = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

def print_header(title: str) -> None:
    print(f"\n=== {title} ===")

def get_os_info() -> None:
    print_header("Operating System")
    print(f"System:     {platform.system()}")
    print(f"Node Name:  {platform.node()}")
    print(f"Release:    {platform.release()}")
    print(f"Version:    {platform.version()}")
    print(f"Machine:    {platform.machine()}")
    print(f"Processor:  {platform.processor()}")

def get_cpu_info() -> None:
    print_header("CPU")
    if psutil:
        phys = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        print(f"Physical cores: {phys}")
        print(f"Logical cores:  {logical}")
        if freq:
            print(f"Max Frequency:  {freq.max:.2f} MHz")
            print(f"Min Frequency:  {freq.min:.2f} MHz")
            print(f"Current Freq.:  {freq.current:.2f} MHz")
        util = psutil.cpu_percent(interval=1)
        print(f"CPU Utilization: {util}%")
    else:
        print("psutil not installed; cannot retrieve detailed CPU info.")
        print(f"Logical cores (fallback): {platform.cpu_count()}")

def get_memory_info() -> None:
    print_header("Memory")
    if psutil:
        vm = psutil.virtual_memory()
        print(f"Total:      {vm.total / (1024**3):.2f} GB")
        print(f"Available:  {vm.available / (1024**3):.2f} GB")
        print(f"Used:       {vm.used / (1024**3):.2f} GB ({vm.percent}%)")
    else:
        print("psutil not installed; cannot retrieve memory info.")

def get_disk_info() -> None:
    print_header("Disk Partitions")
    if psutil:
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                print(f"{part.device} ({part.mountpoint}) — {usage.total / (1024**3):.2f} GB total, {usage.percent}% used")
            except PermissionError:
                continue
    else:
        print("psutil not installed; cannot retrieve disk info.")

def get_gpu_info() -> None:
    print_header("GPU(s)")
    if GPUtil:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPUs found.")
        for gpu in gpus:
            print(f"ID {gpu.id}: {gpu.name}")
            print(f"  Load: {gpu.load * 100:.1f}%  Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
    else:
        print("GPUtil not installed; skipping GPU detection.")

def get_python_env() -> None:
    print_header("Python Environment")
    print(f"Python Version: {platform.python_version()}")
    print(f"Executable:     {sys.executable}")
    print(f"Current Time:   {datetime.now().isoformat()}")
    if pkg_resources:
        installed = list(pkg_resources.working_set)
        print(f"Installed Packages ({len(installed)}):")
        for dist in sorted(installed, key=lambda d: d.project_name.lower()):
            print(f"  {dist.project_name}=={dist.version}")
    else:
        print("pkg_resources not installed; cannot list packages.")
        print("You can run: pip freeze")

def get_relevant_libraries() -> None:
    print_header("Key Libraries Versions")
    libraries = ["numpy", "opencv-python", "torch", "onnx", "psutil"]
    for lib in libraries:
        try:
            # import cv2 for opencv-python
            mod = __import__("cv2") if lib == "opencv-python" else __import__(lib)
            version = getattr(mod, "__version__", "unknown")
            print(f"{lib}: {version}")
        except ImportError:
            print(f"{lib}: not installed")

def main() -> None:
    get_os_info()
    get_cpu_info()
    get_memory_info()
    get_disk_info()
    get_gpu_info()
    get_python_env()
    get_relevant_libraries()

if __name__ == "__main__":
    main()
