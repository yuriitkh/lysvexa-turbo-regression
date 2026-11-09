"""Deterministic, explainable dry-run scheduler; it never starts pytest."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from turbo_regression.admission import RISKY_RESOURCES, requirements_are_clean
from turbo_regression.auto_classification import classify_evidence
from turbo_regression.capabilities import SUPPORTED
from turbo_regression.execution_capacity import allowed_pytest_processes
from turbo_regression.classification import (
    TestClassification,
    classify_test,
    load_manifest,
    stable_test_id,
)

PLAN_VERSION = 1
MODES = frozenset({"FOCUSED", "AFFECTED", "FULL", "FULL_FAST", "QUALIFICATION"})
OUTCOMES = frozenset({
    "SCHEDULED", "UNSCHEDULED", "BLOCKED", "UNCLASSIFIED", "NOT_APPLICABLE",
})
FALLBACK_DURATION_SECONDS = 60.0
FALLBACK_MEMORY_PER_SLOT_BYTES = 512 * 1024 * 1024
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

_CAPABILITY_FIELDS = {
    "pytest": "pytest_available",
    "playwright": "playwright_available",
    "chromium": "chromium_available",
    "docker": "docker_available",
}


class SchedulingError(ValueError):
    pass


def _target_list(value: Sequence[Mapping[str, Any]] | None) -> list[dict[str, str | None]] | None:
    if value is None:
        return None
    result = []
    seen = set()
    for target in value:
        if not isinstance(target, Mapping) or set(target) - {"operating_system", "architecture"}:
            raise SchedulingError("each target must specify operating_system and optional architecture")
        system, architecture = target.get("operating_system"), target.get("architecture")
        if not isinstance(system, str) or system not in {os_name for os_name, _ in SUPPORTED}:
            raise SchedulingError("target operating_system is unsupported")
        if architecture is not None and (
            not isinstance(architecture, str) or (system, architecture) not in SUPPORTED
        ):
            raise SchedulingError("target operating_system/architecture pair is unsupported")
        key = (system, architecture)
        if key in seen:
            raise SchedulingError("duplicate requested target")
        seen.add(key)
        result.append({"operating_system": system, "architecture": architecture})
    if not result:
        raise SchedulingError("requested targets cannot be empty")
    return sorted(result, key=lambda item: (item["operating_system"], item["architecture"] or ""))


def _explicit_serial_group(requirements: Mapping[str, Any]) -> str | None:
    """A declared lock. The synthetic ``unverified-isolation`` bucket is not a lock."""
    serial = requirements.get("serial_group")
    if isinstance(serial, str) and serial:
        return serial
    isolation = requirements.get("isolation_group")
    if isinstance(isolation, str) and isolation:
        return "isolation:" + isolation
    return None


def _risky_resources(isolation) -> list[str]:
    if not isinstance(isolation, list):
        return []
    return sorted({
        str(item.get("resource")) for item in isolation
        if isinstance(item, Mapping) and item.get("resource") in RISKY_RESOURCES
    })


def cross_worker_scopes(isolation) -> list[str]:
    """Scopes that must not run at the same time on any worker.

    A scope without a target, such as ``shared_database``, is the unresolved lock.
    ``shared_database:app.sqlite3`` conflicts with that broad lock and with the same
    path, not with a different path. ``batch_exclusive`` items stay out of this list:
    they only refuse to share a pytest process.
    """
    if not isinstance(isolation, list):
        return []
    scopes = []
    for item in isolation:
        if not isinstance(item, Mapping) or item.get("batch_exclusive") is True:
            continue
        resource = item.get("resource")
        if resource not in RISKY_RESOURCES:
            continue
        scope = item.get("scope")
        if not isinstance(scope, str) or not scope:
            scope = str(resource)
        scopes.append(scope.replace("%", "%25").replace("+", "%2B"))
    return sorted(set(scopes))


def batch_is_exclusive(isolation) -> bool:
    return any(
        isinstance(item, Mapping) and item.get("batch_exclusive") is True
        for item in isolation or []
    )


def _resource_locks(scopes: list[str]) -> list[str]:
    """One lock per proven target. Overlapping scopes then share a lock."""
    return [f"resource:{scope}" for scope in scopes]


def _canonical_resource_group(scopes: list[str]) -> str:
    return "resource:" + "+".join(scopes)


def locks_conflict(left: str, right: str) -> bool:
    """A broad lock conflicts with every narrower lock of the same resource."""
    if left == right:
        return True
    return right.startswith(left + ":") or left.startswith(right + ":")


def _lock_ready(lock: str, group_finish: Mapping[str, float]) -> float:
    ready = group_finish.get(lock, 0.0)
    for other, finish in group_finish.items():
        if locks_conflict(lock, other):
            ready = max(ready, finish)
    return ready


def _provisional_rejection(evidence) -> str | None:
    """FULL_FAST admits an unverified test only when a safe lane is confirmed.

    PARALLEL still requires the Worker's clean-requirements preflight. SERIAL admits a
    declared serial or isolation group, and also an inferred shared resource: each
    resource is its own lock, so two tests that name the same resource cannot run
    together. A test with neither verified isolation, nor a declared lock, nor an
    inferred shared resource stays blocked. Admission is PROVISIONAL: the test is
    never reported as VERIFIED.
    """
    requirements = evidence["requirements"]
    isolation = evidence.get("isolation")
    if evidence["confidence_level"] != "INFERRED" or evidence["status"] != "CLASSIFIED":
        return f"PROVISIONAL_REJECTED: classification is {evidence['confidence_level']}/{evidence['status']}"
    if evidence["blockers"]:
        return "PROVISIONAL_REJECTED: unresolved classification blockers"
    if not isinstance(isolation, list):
        return "PROVISIONAL_REJECTED: isolation evidence is missing"
    if requirements.get("oracle_specific"):
        return "PROVISIONAL_REJECTED: Oracle-specific tests need verified evidence"
    group = _explicit_serial_group(requirements)
    if group or cross_worker_scopes(isolation) or batch_is_exclusive(isolation):
        return None
    if requirements.get("isolation_verified") is False:
        return "PROVISIONAL_REJECTED: isolation constraints need verified evidence"
    if not requirements_are_clean(evidence):
        return "PROVISIONAL_REJECTED: requirements fail the Worker clean-execution preflight"
    return None


def _target_specs(classification, requested_targets):
    allowed = set(classification.allowed_platforms)
    if classification.cross_platform_os:
        specs = []
        for system in classification.cross_platform_os:
            if requested_targets is None:
                options = [{"operating_system": system, "architecture": None}]
            else:
                options = [target for target in requested_targets if target["operating_system"] == system]
            if not options:
                return [], f"mandatory cross-platform target {system} was not requested"
            for target in options:
                if target["architecture"] is None:
                    if not any(os_name == system for os_name, _ in allowed):
                        continue
                elif (system, target["architecture"]) not in allowed:
                    continue
                specs.append(target)
        if not specs:
            return [], "cross-platform targets conflict with declared platform requirements"
        return specs, None
    if requested_targets is None:
        return [{"operating_system": None, "architecture": None}], None
    specs = [
        target for target in requested_targets
        if any(
            os_name == target["operating_system"]
            and (target["architecture"] is None or architecture == target["architecture"])
            for os_name, architecture in allowed
        )
    ]
    if not specs:
        return [], "test requirements do not apply to any requested target"
    return specs, None


def _duration(test_id: str, history: Mapping[str, Any], fallback: float) -> tuple[float, dict]:
    entry = history.get(test_id)
    if isinstance(entry, Mapping):
        duration = entry.get("duration_seconds")
        samples = entry.get("sample_count")
        source_sha = entry.get("source_commit_sha")
        if (
            entry.get("trustworthy") is True
            and isinstance(duration, (int, float)) and not isinstance(duration, bool)
            and math.isfinite(duration) and duration > 0
            and type(samples) is int and samples >= 3
            and isinstance(source_sha, str) and SHA_PATTERN.fullmatch(source_sha)
        ):
            return float(duration), {
                "kind": "history", "sample_count": samples, "source_commit_sha": source_sha,
            }
    return fallback, {"kind": "fallback", "reason": "no trustworthy duration history"}


def _valid_cached_evidence(test_id: str, evidence: Mapping[str, Any]) -> bool:
    requirements = evidence.get("requirements")
    validation = evidence.get("validation")
    if (
        evidence.get("test_id") != test_id
        or evidence.get("confidence_level") not in {"VERIFIED", "INFERRED", "UNKNOWN"}
        or evidence.get("status") not in {"CLASSIFIED", "CONFLICT", "UNKNOWN", "UNCLASSIFIED"}
        or not isinstance(evidence.get("classification_source"), str)
        or not isinstance(evidence.get("reason"), (str, type(None)))
        or not isinstance(evidence.get("confidence"), (int, float))
        or isinstance(evidence.get("confidence"), bool)
        or not math.isfinite(evidence["confidence"])
        or not isinstance(evidence.get("categories"), list)
        or not all(isinstance(item, str) for item in evidence["categories"])
        or not isinstance(evidence.get("blockers"), list)
        or not all(isinstance(item, str) for item in evidence["blockers"])
        or not isinstance(requirements, Mapping)
        or not isinstance(validation, Mapping)
    ):
        return False
    string_lists = ("operating_systems", "architectures", "features", "cross_platform_os")
    if any(
        not isinstance(requirements.get(key), list)
        or not all(isinstance(item, str) for item in requirements[key])
        for key in string_lists
    ):
        return False
    if (
        type(requirements.get("minimum_ram_bytes")) is not int
        or requirements["minimum_ram_bytes"] < 0
        or type(requirements.get("oracle_specific")) is not bool
        or any(
            key in requirements and type(requirements[key]) is not bool
            for key in ("isolation_verified", "qualification")
        )
        or not isinstance(validation.get("required_targets"), list)
        or not all(isinstance(item, str) for item in validation["required_targets"])
    ):
        return False
    return all(
        requirements.get(key) is None or isinstance(requirements[key], str)
        for key in ("docker_architecture", "serial_group", "isolation_group")
    )


def _features_match(capabilities: Mapping[str, Any], required: Sequence[str], docker_arch: str | None) -> bool:
    optional = set(capabilities.get("optional_capabilities", []))
    for feature in required:
        field = _CAPABILITY_FIELDS.get(feature)
        if field is not None:
            if capabilities.get(field) is not True:
                return False
        elif feature not in optional:
            return False
    if "docker" in required:
        expected = docker_arch or capabilities.get("architecture")
        if capabilities.get("docker_architecture") != expected:
            return False
    return True


def _matching_workers(
    devices: Sequence[Mapping[str, Any]],
    classification,
    target: Mapping[str, Any],
    oracle_devices: set[str],
) -> tuple[list[Mapping[str, Any]], list[str], list[Mapping[str, Any]]]:
    candidates = []
    reasons = []
    deferred = []
    for device in devices:
        device_id = device.get("device_id")
        caps = device.get("capabilities")
        if not isinstance(device_id, str) or not isinstance(caps, Mapping):
            continue
        pair = (caps.get("operating_system"), caps.get("architecture"))
        if pair not in classification.allowed_platforms:
            continue
        if device.get("execution_authorized") is not True:
            reasons.append(f"{device_id}: not explicitly trusted for test execution")
            continue
        if target["operating_system"] and pair[0] != target["operating_system"]:
            continue
        if target["architecture"] and pair[1] != target["architecture"]:
            continue
        if classification.oracle_specific and device_id not in oracle_devices:
            reasons.append(f"{device_id}: not explicitly authorized for Oracle")
            continue
        if not _features_match(caps, classification.required_features, classification.docker_architecture):
            reasons.append(f"{device_id}: missing required capabilities")
            continue
        ram = caps.get("available_ram_bytes")
        if type(ram) is not int or ram < classification.minimum_ram_bytes:
            reasons.append(f"{device_id}: insufficient available RAM")
            continue
        if device.get("status") in {"OFFLINE", "BUSY"}:
            deferred.append(device)
            reasons.append(f"{device_id}: worker status is {device.get('status', 'UNKNOWN')}")
            continue
        if device.get("status") != "ONLINE" or device.get("assigned_task") is not None:
            reasons.append(f"{device_id}: worker status is {device.get('status', 'UNKNOWN')}")
            continue
        candidates.append(device)
    candidates.sort(key=lambda device: str(device["device_id"]))
    return candidates, reasons, deferred


def _capacity(device: Mapping[str, Any]) -> int:
    caps = device["capabilities"]
    cores = caps.get("cpu_cores")
    ram = caps.get("available_ram_bytes")
    if type(cores) is not int or cores < 1 or type(ram) is not int or ram < 0:
        return 0
    memory_slots = ram // FALLBACK_MEMORY_PER_SLOT_BYTES
    return max(1, min(cores, memory_slots)) if ram else 0


def _stable_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return prefix + hashlib.sha256(encoded).hexdigest()[:20]


def worker_recommendation_key(test_id: str, target: Mapping[str, Any]) -> str:
    return _stable_id("worker-rec-", {"test_id": test_id, "target": dict(target)})


def _planning_stamp(value: datetime | None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise SchedulingError("planning_time must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def create_plan(
    registry,
    inventory: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    source_commit_sha: str,
    manifest: str | Mapping[str, Any] | None = None,
    validation_manifest: str | Mapping[str, Any] | None = None,
    selected_test_ids: Sequence[str] = (),
    requested_targets: Sequence[Mapping[str, Any]] | None = None,
    oracle_authorized_device_ids: Sequence[str] = (),
    duration_history: Mapping[str, Any] | None = None,
    classification_cache: Mapping[str, Mapping[str, Any]] | None = None,
    worker_recommendations: Mapping[str, str] | None = None,
    fallback_duration_seconds: float = FALLBACK_DURATION_SECONDS,
    planning_time: datetime | None = None,
) -> dict[str, Any]:
    """Create a plan from collected tests and current registry snapshots only."""
    mode = mode.upper() if isinstance(mode, str) else mode
    if mode not in MODES:
        raise SchedulingError(f"unsupported scheduling mode: {mode}")
    if not isinstance(source_commit_sha, str) or not SHA_PATTERN.fullmatch(source_commit_sha):
        raise SchedulingError("source_commit_sha must be a full 40-character Git SHA")
    if not isinstance(fallback_duration_seconds, (int, float)) or isinstance(
        fallback_duration_seconds, bool
    ) or not math.isfinite(fallback_duration_seconds) or fallback_duration_seconds <= 0:
        raise SchedulingError("fallback_duration_seconds must be finite and positive")
    if mode == "FOCUSED" and not selected_test_ids:
        raise SchedulingError("FOCUSED mode requires explicit test IDs")

    manifest_value = load_manifest(manifest)
    targets = _target_list(requested_targets)
    history = duration_history or {}
    if not isinstance(history, Mapping):
        raise SchedulingError("duration_history must be a mapping")
    authorized_oracle = set(oracle_authorized_device_ids)
    if any(not isinstance(item, str) or not item for item in authorized_oracle):
        raise SchedulingError("Oracle authorization entries must be device IDs")

    if mode == "AFFECTED":
        plan = {
            "schema_version": PLAN_VERSION,
            "plan_id": "",
            "source_commit_sha": source_commit_sha,
            "mode": mode,
            "status": "NOT_AVAILABLE",
            "planning_timestamp": _planning_stamp(planning_time),
            "selected_test_inventory": [],
            "test_outcomes": [],
            "batches": [],
            "unscheduled_tests": [],
            "required_os_coverage": {"required": [], "covered": [], "missing": []},
            "statistics": _empty_statistics(),
            "reason": "No reliable changed-code-to-test dependency mapping is configured",
            "fallback_recommendation": "Use FOCUSED with explicit pytest node IDs or FULL",
            "execution_started": False,
        }
        plan["plan_id"] = _stable_id("trp-", {k: v for k, v in plan.items() if k != "planning_timestamp"})
        return plan

    records: dict[str, Mapping[str, Any]] = {}
    for record in inventory:
        if not isinstance(record, Mapping) or not isinstance(record.get("nodeid"), str):
            raise SchedulingError("inventory records must contain pytest node IDs")
        test_id = stable_test_id(record["nodeid"])
        if test_id in records:
            raise SchedulingError(f"duplicate inventory test ID: {test_id}")
        records[test_id] = record

    if mode == "FOCUSED":
        requested = [stable_test_id(test_id) for test_id in selected_test_ids]
        if len(requested) != len(set(requested)):
            raise SchedulingError("FOCUSED test IDs must be unique")
        selected = [(test_id, records.get(test_id)) for test_id in requested]
    else:
        selected = [(test_id, record) for test_id, record in sorted(records.items())]

    devices = registry.list_devices()
    classifications = []
    for test_id, record in selected:
        if record is None:
            classifications.append((test_id, None, "BLOCKED", "TEST_NOT_FOUND_IN_COLLECTION", None))
            continue
        cached_evidence = (classification_cache or {}).get(test_id)
        if (
            isinstance(cached_evidence, Mapping)
            and _valid_cached_evidence(test_id, cached_evidence)
        ):
            evidence = dict(cached_evidence)
        else:
            evidence = classify_evidence(
                record, manifest=manifest_value, validation_manifest=validation_manifest,
                source_commit_sha=source_commit_sha,
            )
        requirements = evidence["requirements"]
        allowed_platforms = tuple(sorted(
            (system, architecture)
            for system, architecture in SUPPORTED
            if (not requirements["operating_systems"]
                or system in requirements["operating_systems"])
            and (not requirements["architectures"]
                 or architecture in requirements["architectures"])
        ))
        classification = TestClassification(
            test_id=test_id,
            status="CLASSIFIED" if evidence["confidence_level"] == "VERIFIED" else evidence["status"],
            categories=tuple(evidence["categories"]),
            allowed_platforms=allowed_platforms,
            required_features=tuple(requirements["features"]),
            docker_architecture=requirements["docker_architecture"],
            minimum_ram_bytes=requirements["minimum_ram_bytes"],
            oracle_specific=requirements["oracle_specific"],
            cross_platform_os=tuple(requirements["cross_platform_os"]),
            serial_group=requirements["serial_group"],
            isolation_group=requirements.get("isolation_group"),
            isolation_verified=requirements.get("isolation_verified", False),
            qualification=requirements.get("qualification", False),
            classification_source=evidence["classification_source"],
            confidence=evidence["confidence"],
            reason="; ".join(evidence["blockers"]) or None,
            fixtures=tuple(record.get("fixtures", [])),
            platform_clues=tuple(record.get("platform_clues", [])),
        )
        classifications.append((test_id, classification, evidence["status"], evidence["reason"], evidence))

    outcomes: list[dict[str, Any]] = []
    jobs = []
    selected_inventory = []
    for test_id, classification, state, reason, evidence in classifications:
        if classification is None:
            outcomes.append({
                "test_id": test_id, "status": "BLOCKED",
                "reasons": ["TEST_NOT_FOUND_IN_COLLECTION"],
            })
            selected_inventory.append({"test_id": test_id, "classification": None})
            continue
        descriptor = evidence
        selected_inventory.append({"test_id": test_id, "classification": descriptor})
        admission = "VERIFIED"
        if evidence["confidence_level"] != "VERIFIED":
            rejection = _provisional_rejection(evidence) if mode == "FULL_FAST" else None
            if mode != "FULL_FAST" or rejection:
                outcome = "BLOCKED" if state == "CONFLICT" or mode == "FULL_FAST" else "UNCLASSIFIED"
                outcomes.append({
                    "test_id": test_id, "status": outcome,
                    "reasons": [
                        *([rejection] if rejection else []),
                        reason or f"classification confidence is {evidence['confidence_level']}",
                    ],
                    "classification": descriptor,
                })
                continue
            admission = "PROVISIONAL"
        if state == "CONFLICT":
            outcomes.append({"test_id": test_id, "status": "BLOCKED", "reasons": [reason]})
            continue
        if mode == "QUALIFICATION" and not classification.qualification:
            outcomes.append({
                "test_id": test_id, "status": "NOT_APPLICABLE",
                "reasons": ["test is not explicitly marked as a qualification check"],
            })
            continue
        if mode != "QUALIFICATION" and classification.qualification and mode == "FOCUSED":
            pass

        specs, spec_error = _target_specs(classification, targets)
        if spec_error:
            outcome = "NOT_APPLICABLE" if "do not apply" in spec_error else "BLOCKED"
            outcomes.append({"test_id": test_id, "status": outcome, "reasons": [spec_error]})
            continue
        duration, duration_source = _duration(
            test_id, history, float(fallback_duration_seconds),
        )
        for target in specs:
            candidates, match_reasons, deferred = _matching_workers(
                devices, classification, target, authorized_oracle,
            )
            if not candidates:
                if not devices:
                    status, reason = "BLOCKED", "NO_REGISTERED_DEVICES"
                elif deferred:
                    status, reason = "UNSCHEDULED", "NO_ONLINE_ELIGIBLE_DEVICE"
                elif match_reasons or (
                    target["operating_system"] and any(
                        (device.get("capabilities") or {}).get("operating_system") == target["operating_system"]
                        for device in devices
                    )
                ):
                    # An unbound target has no platform to miss. Capability, trust and RAM
                    # failures are named in match_reasons and must not be reported as
                    # NO_DEVICE_FOR_REQUIRED_PLATFORM.
                    status, reason = "BLOCKED", "REQUIRED_CAPABILITIES_OR_RESOURCES_UNAVAILABLE"
                else:
                    status, reason = "BLOCKED", "NO_DEVICE_FOR_REQUIRED_PLATFORM"
                outcomes.append({
                    "test_id": test_id, "target": target, "status": status,
                    "reasons": [reason, *match_reasons],
                    "classification": descriptor,
                })
                continue
            # PARALLEL only when isolation is verified, or FULL_FAST already proved the
            # requirements clean. A declared serial/isolation group is the lock for SERIAL.
            # Anything else has no confirmed safe configuration and is not given a synthetic group.
            explicit_group = _explicit_serial_group({
                "serial_group": classification.serial_group,
                "isolation_group": classification.isolation_group,
            })
            scopes = cross_worker_scopes((evidence or {}).get("isolation"))
            exclusive = batch_is_exclusive((evidence or {}).get("isolation"))
            # A declared name and every proven shared target are held. A broad
            # unresolved scope still conflicts with each narrower target.
            if explicit_group or scopes:
                execution_mode = "SERIAL"
                isolated = bool(classification.isolation_verified)
                locks = [explicit_group] if explicit_group else []
                locks.extend(lock for lock in _resource_locks(scopes) if lock not in locks)
                group = explicit_group or _canonical_resource_group(scopes)
            elif classification.isolation_verified or admission == "PROVISIONAL":
                execution_mode = "PARALLEL"
                isolated = True
                group = None
                locks = []
            else:
                outcomes.append({
                    "test_id": test_id, "target": target, "status": "BLOCKED",
                    "reasons": [
                        "NO_CONFIRMED_SAFE_CONFIGURATION: isolation is not verified and no serial group is declared",
                    ],
                    "classification": descriptor,
                })
                continue
            jobs.append({
                "test_id": test_id,
                "classification": classification,
                "target": target,
                "duration": duration,
                "duration_source": duration_source,
                "candidates": candidates,
                "group": group,
                "locks": locks,
                "batch_exclusive": exclusive and execution_mode == "PARALLEL",
                "admission": admission,
                "isolation_verified": isolated,
                "execution_mode": execution_mode,
            })

    # Longest estimated jobs first, with stable IDs breaking ties. Jobs that
    # share an isolation group receive explicit dependency edges below.
    jobs.sort(key=lambda job: (-job["duration"], job["test_id"], job["target"]["operating_system"] or "",
                               job["target"]["architecture"] or ""))
    worker_finish: dict[str, float] = {}
    worker_tail: dict[str, str] = {}
    group_finish: dict[str, float] = {}
    group_tail: dict[str, str] = {}
    group_worker: dict[str, str] = {}
    batches = []
    scheduled_outcomes = []
    for job in jobs:
        choices = []
        for device in job["candidates"]:
            device_id = device["device_id"]
            capacity = _capacity(device)
            if capacity < 1:
                continue
            start = worker_finish.get(device_id, 0.0)
            for lock in job["locks"]:
                start = max(start, _lock_ready(lock, group_finish))
            finish = start + job["duration"]
            platforms = job["classification"].allowed_platforms
            portable = len({architecture for _, architecture in platforms}) > 1 or len({system for system, _ in platforms}) > 1
            capabilities = device["capabilities"]
            # Soft preference: a Linux/ARM64 worker wins a close finish for portable work, and
            # loses when another compatible worker would finish materially sooner.
            # The stored finish stays the real one; only the ranking score is reduced.
            # Duration is not divided by CPU count, otherwise a many-core host would take
            # portable work away from a small worker and undo that preference.
            # Capacity stays a tie-break. Parallel pytest is a per-worker execution limit.
            bonus = job["duration"] * 0.25 if (
                portable
                and capabilities.get("operating_system") == "Linux"
                and capabilities.get("architecture") == "arm64"
            ) else 0.0
            docker_required = "docker" in job["classification"].required_features
            expected_docker = job["classification"].docker_architecture or capabilities.get("architecture")
            docker_rank = 0 if (
                docker_required
                and capabilities.get("operating_system") == "Linux"
                and capabilities.get("architecture") == "arm64"
                and capabilities.get("docker_available") is True
                and capabilities.get("docker_architecture") == expected_docker
            ) else 1
            pinned = {group_worker[lock] for lock in job["locks"] if lock in group_worker}
            group_rank = 0 if len(pinned) == 1 and device_id in pinned else 1
            recommendation_key = worker_recommendation_key(job["test_id"], job["target"])
            recommendation_rank = (
                0 if device_id == (worker_recommendations or {}).get(recommendation_key) else 1
            )
            choices.append((
                group_rank, docker_rank, finish - bonus, recommendation_rank, finish, start, -capacity,
                device_id, device, capacity,
            ))
        if not choices:
            outcomes.append({
                "test_id": job["test_id"], "target": job["target"], "status": "BLOCKED",
                "reasons": ["worker CPU/RAM capacity is unavailable"],
                "classification": job["classification"].as_dict(),
            })
            continue
        _, _, _, _, finish, start, _, device_id, device, capacity = min(choices)
        classification = job["classification"]
        actual_target = {
            "operating_system": device["capabilities"]["operating_system"],
            "architecture": device["capabilities"]["architecture"],
        }
        identity = {
            "source_commit_sha": source_commit_sha,
            "test_id": job["test_id"],
            "target": actual_target,
        }
        batch_id = _stable_id("batch-", identity)
        dependencies = []
        # Worker capacity is enforced by the claim transaction. A worker-tail
        # PASS dependency serialized independent tests and lost all later tests
        # when one assertion failed. Only real isolation ordering belongs here.
        for lock in job["locks"]:
            previous = group_tail.get(lock)
            if previous and previous not in dependencies:
                dependencies.append(previous)
        batch = {
            "batch_id": batch_id,
            "test_ids": [job["test_id"]],
            "target": actual_target,
            "requested_target": job["target"],
            "assigned_worker_id": device_id,
            "eligible_workers": [
                {"device_id": candidate["device_id"], "worker_id": candidate["worker_id"]}
                for candidate in job["candidates"]
            ],
            "estimated_duration_seconds": job["duration"],
            "duration_provenance": job["duration_source"],
            "estimated_start_seconds": round(start, 3),
            "estimated_finish_seconds": round(finish, 3),
            "depends_on": dependencies,
            "dependency_policy": "completion",
            "isolation_group": job["group"],
            "serial_locks": list(job["locks"]),
            "isolation_verified": job["isolation_verified"],
            "execution_mode": job["execution_mode"],
            "batch_exclusive": job["batch_exclusive"],
            "admission": job["admission"],
            "required_capabilities": {
                "operating_systems": sorted({system for system, _ in classification.allowed_platforms}),
                "architectures": sorted({architecture for _, architecture in classification.allowed_platforms}),
                "features": list(classification.required_features),
                "minimum_ram_bytes": classification.minimum_ram_bytes,
                "docker_architecture": classification.docker_architecture,
                "oracle_authorized_environment": classification.oracle_specific,
            },
            "classification_provenance": classification.classification_source,
            "classification_confidence": classification.confidence,
        }
        batches.append(batch)
        worker_finish[device_id] = finish
        worker_tail[device_id] = batch_id
        for lock in job["locks"]:
            group_finish[lock] = finish
            group_tail[lock] = batch_id
            group_worker[lock] = device_id
        scheduled_outcomes.append({
            "test_id": job["test_id"], "target": actual_target,
            "requested_target": job["target"], "status": "SCHEDULED", "admission": job["admission"],
            "assigned_worker_id": device_id, "batch_id": batch_id,
            "classification": classification.as_dict(),
        })

    outcomes.extend(scheduled_outcomes)
    outcomes.sort(key=lambda item: (
        item["test_id"],
        (item.get("target") or {}).get("operating_system") or "",
        (item.get("target") or {}).get("architecture") or "",
        item["status"],
    ))
    batches.sort(key=lambda batch: (batch["estimated_start_seconds"], batch["batch_id"]))

    requested_os = set()
    if targets:
        requested_os.update(target["operating_system"] for target in targets)
    for _, classification, state, _, evidence in classifications:
        if classification and state == "CLASSIFIED":
            requested_os.update(classification.cross_platform_os)
    covered = {
        outcome["target"]["operating_system"] for outcome in outcomes
        if outcome["status"] == "SCHEDULED" and outcome.get("target", {}).get("operating_system")
    }

    statistics = _statistics(batches, devices, requested_os, covered)
    plan = {
        "schema_version": PLAN_VERSION,
        "plan_id": "",
        "source_commit_sha": source_commit_sha,
        "mode": mode,
        "status": "PLANNED",
        "planning_timestamp": _planning_stamp(planning_time),
        "selected_test_inventory": selected_inventory,
        "test_outcomes": outcomes,
        "batches": batches,
        "unscheduled_tests": [item for item in outcomes if item["status"] != "SCHEDULED"],
        "required_os_coverage": {
            "required": sorted(requested_os),
            "covered": sorted(covered),
            "missing": sorted(requested_os - covered),
        },
        "statistics": statistics,
        "admission_summary": _admission_summary(outcomes, selected_inventory),
        "lane_summary": _lane_summary(outcomes, batches, devices),
        "source_preparation": {
            "preparer": "turbo_regression.source_prep.SourcePreparer",
            "source_commit_sha": source_commit_sha,
            "runtime_mixed_with_test_source": False,
        },
        "execution_started": False,
    }
    identity = {key: value for key, value in plan.items() if key != "planning_timestamp"}
    plan["plan_id"] = _stable_id("trp-", identity)
    return plan


def _admission_summary(outcomes, selected_inventory) -> dict[str, Any]:
    """Every selected unique test lands in exactly one bucket; a test with any unplaced target is not scheduled."""
    rows: dict[str, list] = {}
    for outcome in outcomes:
        rows.setdefault(outcome["test_id"], []).append(outcome)
    buckets, reasons = Counter(), Counter()
    for test_id in dict.fromkeys(item["test_id"] for item in selected_inventory):
        test_rows = rows.get(test_id, [])
        unplaced = [row for row in test_rows if row["status"] != "SCHEDULED"]
        if test_rows and not unplaced:
            provisional = any(row.get("admission") == "PROVISIONAL" for row in test_rows)
            buckets["PROVISIONAL" if provisional else "VERIFIED"] += 1
            continue
        statuses = {row["status"] for row in unplaced}
        bucket = (
            "UNSCHEDULED" if statuses == {"UNSCHEDULED"}
            else "NOT_APPLICABLE" if statuses == {"NOT_APPLICABLE"}
            else "BLOCKED"
        )
        buckets[bucket] += 1
        first = unplaced[0] if unplaced else {"reasons": ["MISSING_OUTCOME"]}
        reasons[f"{bucket}: {(first.get('reasons') or [first.get('status')])[0]}"] += 1
    unique = len(dict.fromkeys(item["test_id"] for item in selected_inventory))
    return {
        "unique_tests": unique,
        "verified_scheduled": buckets["VERIFIED"],
        "provisional_scheduled": buckets["PROVISIONAL"],
        "blocked": buckets["BLOCKED"],
        "unscheduled": buckets["UNSCHEDULED"],
        "not_applicable": buckets["NOT_APPLICABLE"],
        "accounted": sum(buckets.values()),
        "reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "verification_scope": "PROVISIONAL" if any(
            row["status"] == "SCHEDULED" and row.get("admission") == "PROVISIONAL" for row in outcomes
        ) else "VERIFIED",
    }


def _lane_summary(outcomes, batches, devices) -> dict[str, Any]:
    """PARALLEL, SERIAL and BLOCKED are mutually exclusive for each scheduled or blocked target row."""
    parallel = [batch for batch in batches if batch.get("execution_mode") == "PARALLEL"]
    serial = [batch for batch in batches if batch.get("execution_mode") == "SERIAL"]
    blocked = [row for row in outcomes if row["status"] == "BLOCKED"]
    reasons = Counter(
        (row.get("reasons") or ["UNSPECIFIED"])[0] for row in blocked
    )
    names = {
        device.get("device_id"): device.get("friendly_name") or device.get("device_id")
        for device in devices if isinstance(device.get("device_id"), str)
    }
    by_worker: dict[str, dict[str, int]] = {}
    for batch in batches:
        worker = batch["assigned_worker_id"]
        lane = batch.get("execution_mode") or "PARALLEL"
        slot = by_worker.setdefault(worker, {"worker": names.get(worker, worker), "PARALLEL": 0, "SERIAL": 0})
        slot[lane] = slot.get(lane, 0) + 1
    return {
        "parallel": len(parallel),
        "serial": len(serial),
        "blocked": len(blocked),
        "blocked_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "workers": dict(sorted(by_worker.items())),
    }


def _empty_statistics() -> dict[str, Any]:
    return {
        "planned_tests_per_worker": {},
        "estimated_total_duration_seconds": 0.0,
        "estimated_completion_time_seconds": 0.0,
        "expected_worker_utilization": {},
        "planned_os_coverage": {"required": [], "covered": [], "missing": []},
        "scheduling_efficiency": 0.0,
        "recommended_pytest_processes": {},
        "actual_duration_seconds": None,
        "retry_metadata": None,
        "reassignment_metadata": None,
        "pass_count": None,
        "fail_count": None,
        "skip_count": None,
    }


def _statistics(batches, devices, requested_os, covered_os) -> dict[str, Any]:
    by_worker: dict[str, list[dict]] = {}
    for batch in batches:
        by_worker.setdefault(batch["assigned_worker_id"], []).append(batch)
    all_ids = sorted({device["device_id"] for device in devices if isinstance(device.get("device_id"), str)})
    counts = {device_id: len(by_worker.get(device_id, [])) for device_id in all_ids}
    completion = max(
        (batch["estimated_finish_seconds"] for batch in batches), default=0.0,
    )
    durations = {
        device_id: sum(batch["estimated_duration_seconds"] for batch in by_worker.get(device_id, []))
        for device_id in all_ids
    }
    active_workers = len(by_worker)
    utilization = {
        device_id: (
            round(durations[device_id] / completion, 4)
            if completion else 0.0
        )
        for device_id in all_ids
    }
    total = sum(durations.values())
    efficiency = min(1.0, total / (completion * active_workers)) if completion and active_workers else 0.0
    processes = {}
    for device in devices:
        device_id = device.get("device_id")
        caps = device.get("capabilities")
        if not isinstance(device_id, str) or not isinstance(caps, Mapping):
            continue
        worker_batches = by_worker.get(device_id, [])
        parallel_safe = bool(worker_batches) and all(
            batch.get("execution_mode") == "PARALLEL" and batch.get("isolation_verified") is True
            for batch in worker_batches
        )
        cores = caps.get("cpu_cores")
        ram = caps.get("available_ram_bytes")
        processes[device_id] = allowed_pytest_processes(
            4,
            cpu_cores=cores if type(cores) is int else 1,
            available_ram_bytes=ram if type(ram) is int else 0,
            parallel_safe=parallel_safe,
        )
    return {
        "planned_tests_per_worker": counts,
        "estimated_total_duration_seconds": round(total, 3),
        "estimated_completion_time_seconds": round(completion, 3),
        "expected_worker_utilization": utilization,
        "planned_os_coverage": {
            "required": sorted(requested_os),
            "covered": sorted(covered_os),
            "missing": sorted(set(requested_os) - set(covered_os)),
        },
        "scheduling_efficiency": round(efficiency, 4),
        "recommended_pytest_processes": processes,
        "actual_duration_seconds": None,
        "retry_metadata": None,
        "reassignment_metadata": None,
        "pass_count": None,
        "fail_count": None,
        "skip_count": None,
    }
