"""How many pytest processes one Worker may run at once, and when it should take no new job.

The safe default is one process. A higher configured maximum is still reduced when RAM is
tight, the host has few CPUs, or any test in the batch is not allowed to run beside another.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

# One pytest process stays well under this reserve once the suite import is shared. Two reserves must fit in available RAM before a second process starts.
PROCESS_RESERVE_BYTES = 512 * 1024 * 1024
# Below this, even a configured second process stays off.
PRESSURE_RAM_BYTES = 1536 * 1024 * 1024
# A batch needs a checkout plus pytest scratch. One gibibyte is the floor on every filesystem it writes.
MIN_FREE_DISK_BYTES = 1024 ** 3
# Local JUnit and batch logs are copies. The signed receipt already left the machine; do not let the copies fill the disk.
ARTIFACT_BUDGET_BYTES = 512 * 1024 * 1024
SCRATCH_PREFIX = "turbo-batch-"
MAX_PROCESSES = 4


def assignments_are_parallel_safe(assignments: Sequence[Mapping]) -> bool:
    """Independent verified tests only. A shared isolation group, an unverified test, or a browser/docker feature stays serial."""
    for assignment in assignments:
        if assignment.get("isolation_verified") is not True:
            return False
        if assignment.get("isolation_group"):
            return False
        required = assignment.get("required_capabilities")
        features = required.get("features") if isinstance(required, Mapping) else None
        if features:
            return False
    return True


def allowed_pytest_processes(
    requested: int,
    *,
    cpu_cores: int,
    available_ram_bytes: int,
    parallel_safe: bool,
) -> int:
    """The number of pytest processes to start. Never above the request, and 1 whenever that is safer."""
    if type(requested) is not int or not 1 <= requested <= MAX_PROCESSES:
        raise ValueError(f"max pytest processes must be between 1 and {MAX_PROCESSES}")
    if requested == 1 or not parallel_safe:
        return 1
    if type(cpu_cores) is not int or cpu_cores < 1:
        return 1
    if type(available_ram_bytes) is not int or available_ram_bytes < PRESSURE_RAM_BYTES:
        return 1
    ram_slots = available_ram_bytes // PROCESS_RESERVE_BYTES
    # Leave half the logical CPUs for the operating system and everything else on the machine.
    cpu_slots = max(1, cpu_cores // 2)
    return max(1, min(requested, cpu_slots, ram_slots, MAX_PROCESSES))


def cgroup_memory_available() -> int | None:
    """Bytes this process may still allocate under its own cgroup, when Linux exposes one."""
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    relative = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            relative = parts[2]
    if not relative:
        return None
    base = Path("/sys/fs/cgroup") / relative.lstrip("/")
    try:
        limit = (base / "memory.max").read_text(encoding="ascii").strip()
        if limit == "max":
            return None
        current = int((base / "memory.current").read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return max(0, int(limit) - current)


def claims_allowed(*, available_ram_bytes, available_disk_bytes, cgroup_available_bytes=None) -> bool:
    """False when the host or this Worker's cgroup has no room for another test. CPU load is not a reason."""
    if type(available_ram_bytes) is not int or available_ram_bytes < PRESSURE_RAM_BYTES:
        return False
    if cgroup_available_bytes is not None and (
        type(cgroup_available_bytes) is not int or cgroup_available_bytes < PROCESS_RESERVE_BYTES
    ):
        return False
    if type(available_disk_bytes) is not int or available_disk_bytes < MIN_FREE_DISK_BYTES:
        return False
    return True


def free_bytes(path: str | Path) -> int:
    """Free bytes on the filesystem that holds `path` (or its nearest existing parent)."""
    candidate = Path(path)
    probe = candidate if candidate.exists() else candidate.parent
    if not probe.exists():
        raise OSError(f"cannot measure disk space for {path}")
    return int(shutil.disk_usage(probe).free)


def tightest_free_bytes(paths: Sequence[str | Path]) -> int:
    """The smallest free space among the filesystems a Worker will actually write.

    A host can have tens of gigabytes free while a separate scratch or tmp quota is full.
    A small tmpfs scratch directory can be full while the host as a whole still reports plenty of space.
    """
    measured = [free_bytes(path) for path in paths]
    if not measured:
        raise OSError("no filesystem to measure")
    return min(measured)


def scratch_has_room(paths: Sequence[str | Path], *, minimum: int = MIN_FREE_DISK_BYTES) -> bool:
    try:
        return tightest_free_bytes(paths) >= minimum
    except OSError:
        return False


def reclaim_stale_scratch(directory: str | Path, *, keep: Path) -> int:
    """Remove scratch directories left by a Worker that died mid-batch. The live one is kept."""
    root = Path(directory)
    if not root.is_dir():
        return 0
    kept = keep.resolve()
    removed = 0
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(SCRATCH_PREFIX) or child.resolve() == kept:
            continue
        shutil.rmtree(child, ignore_errors=True)
        if not child.exists():
            removed += 1
    return removed


def prune_artifact_copies(directory: str | Path, *, keep: set[Path], budget_bytes: int = ARTIFACT_BUDGET_BYTES) -> int:
    """Delete the oldest local JUnit and batch-log copies once they pass the budget. Receipts are not stored here."""
    root = Path(directory)
    if not root.is_dir() or budget_bytes < 1:
        return 0
    kept = {path.resolve() for path in keep}
    copies = [
        path for path in root.iterdir()
        if path.is_file() and (path.name.endswith(".junit.xml") or path.name.endswith(".batch.log"))
        and path.resolve() not in kept
    ]
    copies.sort(key=lambda path: path.stat().st_mtime)
    total = sum(path.stat().st_size for path in copies) + sum(
        path.stat().st_size for path in kept if path.is_file()
    )
    removed = 0
    for path in copies:
        if total <= budget_bytes:
            break
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        if not path.exists():
            total -= size
            removed += 1
    return removed
