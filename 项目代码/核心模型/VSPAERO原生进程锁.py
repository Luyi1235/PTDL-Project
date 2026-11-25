"""Serialize native OpenVSP lifetimes across workbench and research processes."""

from contextlib import contextmanager
from functools import wraps
import os


@contextmanager
def native_solver_lock(timeout_s=1800):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel.ReleaseMutex.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateMutexW(None, False, "Local\\PTDL_OpenVSP_Native")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = False
        try:
            status = kernel.WaitForSingleObject(handle, 0)
            if status == 0x102:
                print("Waiting for shared VSPAERO native solver slot", flush=True)
                status = kernel.WaitForSingleObject(handle, int(timeout_s*1000))
            if status not in (0, 0x80):
                if status == 0x102:
                    raise TimeoutError("VSPAERO native solver queue timeout")
                raise ctypes.WinError(ctypes.get_last_error())
            acquired = True
            yield
        finally:
            if acquired:
                kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
        return
    import fcntl
    from pathlib import Path
    import tempfile
    import time
    path = Path(tempfile.gettempdir()) / f"ptdl_vspaero_{os.getuid()}.lock"
    with path.open("a+b") as stream:
        deadline = time.monotonic()+timeout_s
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("VSPAERO native solver queue timeout")
                time.sleep(.25)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def serialized_native_solver(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with native_solver_lock():
            return function(*args, **kwargs)
    return wrapped
