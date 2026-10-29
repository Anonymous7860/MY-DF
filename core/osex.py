import os
import sys

if sys.platform[0:3] == 'win':
    from ctypes import windll
    from ctypes import wintypes

PROCESS_PRIORITIES = ('idle', 'below_normal', 'normal')
_WINDOWS_PRIORITY_CLASSES = {
    'idle': 0x00000040,
    'below_normal': 0x00004000,
    'normal': 0x00000020,
}


def normalize_process_priority(priority):
    normalized = str(priority or 'idle').strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'low': 'idle',
        'lowest': 'idle',
        'belownormal': 'below_normal',
        'default': 'normal',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROCESS_PRIORITIES:
        return 'idle'
    return normalized


def set_process_priority(priority):
    """Apply a validated process priority and return its canonical name."""
    normalized = normalize_process_priority(priority)
    try:
        if sys.platform[0:3] == 'win':
            GetCurrentProcess = windll.kernel32.GetCurrentProcess
            GetCurrentProcess.restype = wintypes.HANDLE
            SetPriorityClass = windll.kernel32.SetPriorityClass
            SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            SetPriorityClass.restype = wintypes.BOOL
            if not SetPriorityClass(
                GetCurrentProcess(),
                _WINDOWS_PRIORITY_CLASSES[normalized],
            ):
                raise OSError("SetPriorityClass failed")
        elif 'darwin' in sys.platform:
            if normalized == 'idle':
                os.nice(20)
            elif normalized == 'below_normal':
                os.nice(10)
        elif 'linux' in sys.platform:
            if normalized == 'idle':
                os.nice(20)
            elif normalized == 'below_normal':
                os.nice(10)
    except Exception as error:
        print(f"Unable to set process priority to {normalized}: {error}")
    return normalized


def set_process_lowest_prio():
    return set_process_priority('idle')

def set_process_dpi_aware():
    if sys.platform[0:3] == 'win':
        windll.user32.SetProcessDPIAware(True)

def get_screen_size():
    if sys.platform[0:3] == 'win':
        user32 = windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    elif 'darwin' in sys.platform:
        pass
    elif 'linux' in sys.platform:
        pass
        
    return (1366, 768)
        
