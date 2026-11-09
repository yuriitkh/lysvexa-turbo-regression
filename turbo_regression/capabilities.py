"""Local, non-executing worker capability discovery."""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


SUPPORTED = {("Windows", "x64"), ("Linux", "x64"), ("Linux", "arm64")}


def normalize_architecture(value: str) -> str:
    lowered = value.lower()
    if lowered in {"amd64", "x86_64", "x64"}:
        return "x64"
    if lowered in {"aarch64", "arm64"}:
        return "arm64"
    return lowered


def _available_memory(system: str) -> int:
    if system == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    try:
        available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as error:
        raise OSError("Unable to determine available RAM") from error
    if available < 0:
        raise OSError("Unable to determine available RAM")
    return available


def _chromium_available(system: str, home: Path) -> bool:
    if not importlib.util.find_spec("playwright"):
        return False
    if any(shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "chrome")):
        return True
    roots = []
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured and configured != "0":
        roots.append(Path(configured).expanduser())
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.append(Path(local) / "ms-playwright")
    else:
        roots.append(home / ".cache" / "ms-playwright")
    executables = (
        Path("chrome-win") / "chrome.exe",
        Path("chrome-win64") / "chrome.exe",
        Path("chrome-linux") / "chrome",
        Path("chrome-linux64") / "chrome",
    )
    return any(
        (browser / executable).is_file()
        for root in roots
        if root.is_dir()
        for browser in root.glob("chromium-*")
        for executable in executables
    )


def _docker(system_architecture: str) -> tuple[bool, str | None]:
    if not shutil.which("docker"):
        return False, None
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Architecture}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return False, None
    if result.returncode:
        return False, None
    architecture = normalize_architecture(result.stdout.strip())
    return bool(architecture), architecture or None


def detect_capabilities(*, system: str | None = None, machine: str | None = None,
                        home: Path | None = None) -> dict:
    """Report facts about this host; never launches tests or a browser."""
    operating_system = system or platform.system()
    architecture = normalize_architecture(machine or platform.machine())
    if (operating_system, architecture) not in SUPPORTED:
        raise ValueError(f"Unsupported worker platform: {operating_system} {architecture}")

    root = home or Path.home()
    from turbo_regression.execution_capacity import tightest_free_bytes
    disk_paths = [root, Path(tempfile.gettempdir())]
    docker_available, docker_architecture = _docker(architecture)
    optional = []
    if operating_system == "Windows":
        optional.append("windows_api")
    if shutil.which("pwsh") or (operating_system == "Windows" and shutil.which("powershell")):
        optional.append("powershell")
    if operating_system == "Linux" and Path("/run/systemd/system").is_dir():
        optional.append("systemd")
    if architecture == "arm64":
        optional.append("arm64")
    if docker_available and docker_architecture == architecture:
        optional.append("native_docker")
    return {
        "operating_system": operating_system,
        "architecture": architecture,
        "cpu_cores": os.cpu_count() or 1,
        "available_ram_bytes": _available_memory(operating_system),
        "available_disk_bytes": tightest_free_bytes(disk_paths),
        "python_version": platform.python_version(),
        "pytest_available": importlib.util.find_spec("pytest") is not None,
        "playwright_available": importlib.util.find_spec("playwright") is not None,
        "chromium_available": _chromium_available(operating_system, root),
        "docker_available": docker_available,
        "docker_architecture": docker_architecture,
        "optional_capabilities": sorted(optional),
    }
