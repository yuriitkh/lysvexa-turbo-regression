"""Admission rules the scheduler shares with a worker-side preflight.

Both items are the real implementation, extracted unchanged in behaviour from two larger modules of the full system that this showcase does not
include (a validation module and a remote-execution module). They are small and self-contained, so they live here instead of being stubbed.
"""
from __future__ import annotations

from typing import Any, Mapping

# Shared resources a test must not hold concurrently with another test touching the same resource.
RISKY_RESOURCES = frozenset({
    "browser_profile", "docker_resource", "external_service", "filesystem_lock",
    "fixed_tcp_port", "global_environment", "shared_database",
})


def requirements_are_clean(analysis: Mapping[str, Any]) -> bool:
    """A test whose requirements can be re-checked from source without a committed isolation stamp.

    It refuses conflicts, blockers, risky shared resources, environment-restricted tests and serial or isolation groups,
    unless the committed classification itself says isolation was verified (`isolation_verified` is not False).
    """
    requirements = analysis.get("requirements") if isinstance(analysis.get("requirements"), Mapping) else {}
    risky = any(
        item.get("resource") in RISKY_RESOURCES
        for item in analysis.get("isolation") or []
        if isinstance(item, Mapping)
    )
    return (
        analysis.get("status") in {"CLASSIFIED", "UNCLASSIFIED"}
        and not analysis.get("blockers")
        and not risky
        and requirements.get("isolation_verified") is not False
        and not requirements.get("oracle_specific")
        and not requirements.get("serial_group")
        and not requirements.get("isolation_group")
    )
