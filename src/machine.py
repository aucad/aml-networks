import psutil
import platform


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def machine_details():
    result = {}
    uname = platform.uname()
    svmem = psutil.virtual_memory()
    result["system"] = uname.system
    result["release"] = uname.release
    result["version"] = uname.version
    result["machine"] = uname.machine
    result["processor"] = uname.processor
    result["physical_cores"] = psutil.cpu_count(logical=False)
    result["total_cores"] = psutil.cpu_count(logical=True)
    try:
        cpufreq = psutil.cpu_freq()
        result["max_frequency"] = cpufreq.max
        result["min_frequency"] = cpufreq.min
        result["current_frequency"] = cpufreq.current
    except FileNotFoundError:
        pass
    result["cpu_usage_per_core"] = \
        psutil.cpu_percent(percpu=True, interval=1)
    result["total_cpu_usage"] = psutil.cpu_percent()
    result["total"] = get_size(svmem.total)
    result["available"] = get_size(svmem.available)
    result["used"] = get_size(svmem.used)
    result["percentage"] = svmem.percent
    return result
