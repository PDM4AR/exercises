import resource
import platform
from typing import Optional


class MemoryLimitExceededException(Exception):
    pass


def _get_memory():
    """Free memory in kB, works on Linux only"""
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def set_memory_limit(
    mode: int = 0,
    # memory limit on evaluation server: 8 GB
    abs_mem_bytes: int = 8 * ((2 ** 10) ** 3)  ,
    percentage: Optional[float] = None,
) -> None:
    """
    Works on Linux only

    When testing locally on non-Linux systems, other monitoring should be used

    Parameters:
        mode = 0: absolute mode | 1: percentage mode
        abs_mem_bytes: absolute memory allowed in bytes
        percentage: percentage allowed, of the free memory
    """
    if platform.system() != "Linux":
        print('Memory limiting only works on linux!')
        return

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        assert mode in {0, 1}, f"Invalid mode {mode}"
        if mode == 0:
            assert hard == -1 or hard >= abs_mem_bytes, "Invalid memory bound"
            resource.setrlimit(resource.RLIMIT_AS, (abs_mem_bytes, hard))
        elif mode == 1:
            assert percentage is not None and 0.0 <= percentage <= 1.0
            resource.setrlimit(
                resource.RLIMIT_AS, (_get_memory() * 1024 * percentage, hard))
    except Exception as e:
        print(f"No effect setting memory limit. Details: {e}")
