"""Plan a small synthetic test suite for two synthetic workers.

Nothing is executed, nothing touches the network, and no real host is involved: the workers are rows in a throw-away SQLite file, the test IDs are
made up, and the "validation runs" are generated here so the output is deterministic.

    python examples/demo.py
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # run from a plain checkout, no installation needed

from turbo_regression.auto_classification import classify_evidence
from turbo_regression.redaction import redact
from turbo_regression.registry import DeviceRegistry
from turbo_regression.scheduler import create_plan

COMMIT = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
GIB = 1024 ** 3

TEST_ADD = "tests/test_math.py::test_add"
TEST_WINDOWS = "tests/test_paths.py::test_drive_letters"
TEST_SYSTEMD = "tests/test_service.py::test_unit_text"
TEST_SERIAL = "tests/test_store.py::test_shared_file_database"
TEST_UNKNOWN = "tests/test_new.py::test_without_metadata"

# What a pytest collection step would hand to the planner (minimal records).
INVENTORY = [
    {"nodeid": nodeid, "fixtures": [], "markers": [], "platform_clues": []}
    for nodeid in (TEST_ADD, TEST_WINDOWS, TEST_SYSTEMD, TEST_SERIAL, TEST_UNKNOWN)
]

# Explicit, reviewed scheduling metadata. TEST_UNKNOWN deliberately has none.
MANIFEST = {
    "schema_version": 1,
    "tests": {
        TEST_ADD: {"verified": True, "isolation_verified": True},
        TEST_WINDOWS: {"verified": True, "operating_systems": ["Windows"], "isolation_verified": True},
        TEST_SYSTEMD: {"verified": True, "operating_systems": ["Linux"], "architectures": ["arm64"],
                       "features": ["systemd"], "isolation_verified": True},
        TEST_SERIAL: {"verified": True, "serial_group": "file-database"},
    },
}


def capabilities(system: str, architecture: str, cores: int, ram_gib: int, optional: list[str]) -> dict:
    return {
        "operating_system": system, "architecture": architecture, "cpu_cores": cores,
        "available_ram_bytes": ram_gib * GIB, "available_disk_bytes": 50 * GIB, "python_version": "3.12.0",
        "pytest_available": True, "playwright_available": False, "chromium_available": False,
        "docker_available": False, "docker_architecture": None, "optional_capabilities": sorted(optional),
    }


WORKERS = {
    "windows": ("demo-windows", "Synthetic Windows worker", capabilities("Windows", "x64", 8, 16, ["powershell", "windows_api"])),
    "arm": ("demo-arm", "Synthetic Linux ARM64 worker", capabilities("Linux", "arm64", 2, 8, ["arm64", "systemd"])),
}


def build_registry(directory: Path, *, trusted: bool = True, names: tuple[str, ...] = ("windows", "arm")):
    """A registry with the chosen synthetic workers enrolled. `trusted=False` leaves them without execution authorization."""
    registry = DeviceRegistry(Path(directory) / "registry.sqlite3", clock=lambda: NOW)
    enrolled = {}
    for name in names:
        worker_id, friendly_name, caps = WORKERS[name]
        token = registry.issue_enrollment_token(worker_id=worker_id)
        enrolled[name] = registry.enroll(token, friendly_name, caps, worker_id=worker_id)
        if trusted:
            registry.set_execution_authorized(enrolled[name]["device_id"], True)
    return registry, enrolled


def validation_evidence(commit: str = COMMIT) -> dict:
    """The record a validation run would leave: every platform a test claims, validated at `commit`, bound to the test's current requirements."""
    entries = {}
    for record in INVENTORY:
        if record["nodeid"] not in MANIFEST["tests"]:
            continue
        analysis = classify_evidence(record, manifest=MANIFEST, source_commit_sha=commit)
        entries[record["nodeid"]] = {
            "requirements_sha256": analysis["requirements_sha256"],
            "targets": {
                target: {"result": "PASS", "source_commit_sha": commit, "run_id": "synthetic-run-1", "evidence": "synthetic evidence"}
                for target in analysis["validation"]["required_targets"]
            },
        }
    return {"schema_version": 1, "validations": entries}


def make_plan(registry, *, validation=None, mode: str = "FULL", inventory=None, commit: str = COMMIT):
    return create_plan(
        registry, INVENTORY if inventory is None else inventory,
        mode=mode, source_commit_sha=commit, manifest=MANIFEST, validation_manifest=validation, planning_time=NOW,
    )


def summarize(plan: dict, workers: dict) -> list[str]:
    names = {worker["device_id"]: name for name, worker in workers.items()}
    lines = []
    for outcome in plan["test_outcomes"]:
        if outcome["status"] == "SCHEDULED":
            lines.append(f"{outcome['test_id']:<50} SCHEDULED    -> {names[outcome['assigned_worker_id']]}")
        else:
            lines.append(f"{outcome['test_id']:<50} {outcome['status']:<12} {outcome['reasons'][0]}")
    return lines


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="turbo-showcase-") as directory:
        registry, workers = build_registry(Path(directory))
        print("Plan with recorded validation evidence:")
        plan = make_plan(registry, validation=validation_evidence())
        print("\n".join("  " + line for line in summarize(plan, workers)))
        print(f"  batches: {len(plan['batches'])}, execution_started: {plan['execution_started']}")

        print("\nSame suite, same workers, but no validation evidence (nothing may be guessed):")
        bare = make_plan(registry, validation=None)
        print("\n".join("  " + line for line in summarize(bare, workers)))
        print(f"  batches: {len(bare['batches'])}")

        print("\nLogging never prints a credential:")
        print("  " + redact(f"worker enrolled, credential={workers['windows']['credential']}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
