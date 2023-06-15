"""
Capture details of the current machine and runtime.
"""

import psutil
import platform
import sys


def get_size(bytes_, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes_ < factor:
            return f"{bytes_:.2f}{unit}{suffix}"
        bytes_ /= factor


def machine_details():
    result = {}
    uname = platform.uname()
    mem = psutil.virtual_memory()
    result["system"] = uname.system
    result["release"] = uname.release
    result["version"] = uname.version
    result["machine"] = uname.machine
    result["processor"] = uname.processor
    result["cpu_physical_cores"] = psutil.cpu_count(logical=False)
    result["cpu_total_cores"] = psutil.cpu_count(logical=True)
    try:
        cpufreq = psutil.cpu_freq()
        result["cpu_max_frequency"] = cpufreq.max
        result["cpu_min_frequency"] = cpufreq.min
        result["cpu_current_frequency"] = cpufreq.current
    except FileNotFoundError:
        pass
    result["cpu_usage_per_core"] = \
        psutil.cpu_percent(percpu=True, interval=1)
    result["cpu_total_usage"] = psutil.cpu_percent()
    result["virtual_mem_total_size"] = get_size(mem.total)
    result["virtual_mem_available"] = get_size(mem.available)
    result["virtual_mem_used"] = get_size(mem.used)
    result["virtual_mem_percentage"] = mem.percent
    result["python_runtime"] = sys.version
    return result
