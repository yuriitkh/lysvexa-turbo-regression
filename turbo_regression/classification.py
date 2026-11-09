"""Conservative pytest metadata classification for Turbo Regression plans."""
from __future__ import annotations

from dataclasses import dataclass
import json
import fnmatch
import glob
import re
from pathlib import PurePosixPath
from typing import Any, Mapping

from turbo_regression.capabilities import SUPPORTED

SUPPORTED_OS = frozenset(system for system, _ in SUPPORTED)
SUPPORTED_ARCHITECTURES = frozenset(architecture for _, architecture in SUPPORTED)
FEATURES = frozenset({
    "pytest", "playwright", "chromium", "docker", "windows_api", "powershell", "systemd",
    "native_docker",
})
MARKER_NAMES = frozenset({
    "turbo", "portable", "windows_only", "linux_only", "arm64_native", "browser", "docker",
    "oracle_specific", "cross_platform", "serial_only", "qualification",
})


class ClassificationError(ValueError):
    pass


@dataclass(frozen=True)
class TestClassification:
    test_id: str
    status: str
    categories: tuple[str, ...]
    allowed_platforms: tuple[tuple[str, str], ...]
    required_features: tuple[str, ...]
    docker_architecture: str | None
    minimum_ram_bytes: int
    oracle_specific: bool
    cross_platform_os: tuple[str, ...]
    serial_group: str | None
    isolation_group: str | None
    isolation_verified: bool
    qualification: bool
    classification_source: str
    confidence: float
    reason: str | None
    fixtures: tuple[str, ...]
    platform_clues: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "status": self.status,
            "categories": list(self.categories),
            "requirements": {
                "allowed_platforms": [
                    {"operating_system": system, "architecture": architecture}
                    for system, architecture in self.allowed_platforms
                ],
                "required_features": list(self.required_features),
                "docker_architecture": self.docker_architecture,
                "minimum_ram_bytes": self.minimum_ram_bytes,
                "oracle_authorized_environment": self.oracle_specific,
                "cross_platform_os": list(self.cross_platform_os),
                "serial_group": self.serial_group,
                "isolation_group": self.isolation_group,
                "isolation_verified": self.isolation_verified,
                "qualification": self.qualification,
            },
            "classification_source": self.classification_source,
            "confidence": self.confidence,
            "reason": self.reason,
            "fixtures": list(self.fixtures),
            "platform_clues": list(self.platform_clues),
        }


def load_manifest(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        manifest: Any = {"schema_version": 1, "tests": {}, "fixture_capabilities": {}}
    elif isinstance(value, str):
        with open(value, encoding="utf-8") as stream:
            manifest = json.load(stream)
    else:
        manifest = dict(value)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ClassificationError("classification manifest must use schema_version 1")
    if set(manifest) - {"schema_version", "tests", "rules", "fixture_capabilities"}:
        raise ClassificationError("classification manifest has unknown top-level fields")
    tests = manifest.get("tests", {})
    rules = manifest.get("rules", {})
    fixtures = manifest.get("fixture_capabilities", {})
    if not isinstance(tests, dict) or not isinstance(rules, dict) or not isinstance(fixtures, dict):
        raise ClassificationError("manifest tests, rules, and fixture_capabilities must be objects")
    for selector, metadata in tests.items():
        if not isinstance(selector, str) or not selector or not isinstance(metadata, dict):
            raise ClassificationError("manifest tests must map stable selectors to metadata objects")
    for selector, metadata in rules.items():
        if not isinstance(selector, str) or not selector or not isinstance(metadata, dict):
            raise ClassificationError("manifest rules must map selectors to metadata objects")
    for fixture, capabilities in fixtures.items():
        if not isinstance(fixture, str) or not isinstance(capabilities, list):
            raise ClassificationError("fixture_capabilities must map fixture names to feature lists")
        if any(feature not in FEATURES for feature in capabilities):
            raise ClassificationError(f"unknown capability configured for fixture {fixture}")
    return {
        "schema_version": 1, "tests": tests, "rules": rules,
        "fixture_capabilities": fixtures,
    }


def _metadata(test: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    nodeid = test.get("nodeid")
    if not isinstance(nodeid, str) or not nodeid:
        raise ClassificationError("collected test is missing its stable nodeid")
    overrides = manifest["tests"]
    if nodeid in overrides:
        return overrides[nodeid], "manifest:" + nodeid
    escaped = glob.escape(nodeid)         # an exact-id selector for an id with glob characters is stored escaped
    if escaped != nodeid and escaped in overrides:
        return overrides[escaped], "manifest:" + escaped
    path = nodeid.split("::", 1)[0].replace("\\", "/")
    if path in overrides:
        return overrides[path], "manifest:" + path
    for selector, metadata in sorted(manifest.get("rules", {}).items()):
        if fnmatch.fnmatchcase(nodeid, selector):
            return metadata, "manifest-rule:" + selector

    fixtures = test.get("fixtures", [])
    mapped_features = sorted({
        feature
        for fixture in fixtures
        for feature in manifest["fixture_capabilities"].get(fixture, [])
    })
    values: list[dict[str, Any]] = []
    for marker in test.get("markers", []):
        if not isinstance(marker, dict) or marker.get("name") not in MARKER_NAMES:
            continue
        name = marker["name"]
        args = marker.get("args", [])
        kwargs = marker.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ClassificationError(f"invalid {name} marker metadata on {nodeid}")
        if name == "turbo":
            values.append(dict(kwargs))
        elif name == "portable":
            values.append({"operating_systems": sorted(SUPPORTED_OS)})
        elif name == "windows_only":
            values.append({"operating_systems": ["Windows"]})
        elif name == "linux_only":
            values.append({"operating_systems": ["Linux"]})
        elif name == "arm64_native":
            values.append({"operating_systems": ["Linux"], "architectures": ["arm64"]})
        elif name == "browser":
            values.append({"features": ["playwright", "chromium"]})
        elif name == "docker":
            values.append({"features": ["docker"]})
        elif name == "oracle_specific":
            values.append({"oracle_specific": True})
        elif name == "cross_platform":
            values.append({"cross_platform_os": kwargs.get("operating_systems", args)})
        elif name == "serial_only":
            values.append({"serial_group": kwargs.get("group", "global")})
        elif name == "qualification":
            values.append({"qualification": True})
    if not values:
        if mapped_features:
            return {"features": mapped_features, "verified": True}, "fixture-capability-map"
        return None, "pytest-collection"
    if len(values) == 1:
        return values[0], "pytest-marker"
    combined: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if key in {"operating_systems", "architectures"}:
                previous = set(combined.get(key, item))
                combined[key] = sorted(previous.intersection(item))
            elif key == "features":
                combined[key] = sorted(set(combined.get(key, [])).union(item))
            elif key in combined and combined[key] != item:
                raise ClassificationError(f"conflicting {key} markers on {nodeid}")
            else:
                combined[key] = item
    return combined, "pytest-markers"


def _string_list(value: Any, key: str, allowed: frozenset[str] | None = None) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ClassificationError(f"{key} must be a list of strings")
    if len(value) != len(set(value)) or (allowed is not None and not set(value) <= allowed):
        raise ClassificationError(f"{key} contains duplicates or unsupported values")
    return value


def _platforms(metadata: Mapping[str, Any]) -> set[tuple[str, str]]:
    operating_systems = _string_list(
        metadata.get("operating_systems", sorted(SUPPORTED_OS)),
        "operating_systems", SUPPORTED_OS,
    )
    architectures = _string_list(
        metadata.get("architectures", sorted(SUPPORTED_ARCHITECTURES)),
        "architectures", SUPPORTED_ARCHITECTURES,
    )
    return {
        (system, architecture) for system, architecture in SUPPORTED
        if system in operating_systems and architecture in architectures
    }


def classify_test(
    test: Mapping[str, Any],
    manifest_value: str | Mapping[str, Any] | None = None,
) -> TestClassification:
    manifest = load_manifest(manifest_value)
    metadata, source = _metadata(test, manifest)
    fixtures = tuple(sorted(set(test.get("fixtures", []))))
    clues = tuple(sorted(set(test.get("platform_clues", []))))

    if metadata is None:
        details = []
        if fixtures:
            details.append("fixture dependencies require explicit capability mapping")
        if clues:
            details.append("test contains host-platform checks")
        if any(marker.get("name") == "skipif" for marker in test.get("markers", [])):
            details.append("pytest skipif condition is not sufficient proof of platform eligibility")
        reason = "No verified scheduling metadata"
        if details:
            reason += "; " + "; ".join(details)
        return TestClassification(
            test_id=test["nodeid"], status="UNCLASSIFIED", categories=("UNCLASSIFIED",),
            allowed_platforms=(), required_features=(), docker_architecture=None,
            minimum_ram_bytes=0, oracle_specific=False, cross_platform_os=(),
            serial_group=None, isolation_group=None, isolation_verified=False,
            qualification=False, classification_source=source, confidence=0.0,
            reason=reason, fixtures=fixtures, platform_clues=clues,
        )

    if source == "fixture-capability-map" and (
        clues or any(marker.get("name") == "skipif" for marker in test.get("markers", []))
    ):
        return TestClassification(
            test_id=test["nodeid"], status="UNCLASSIFIED", categories=("UNCLASSIFIED",),
            allowed_platforms=(), required_features=tuple(metadata["features"]),
            docker_architecture=None, minimum_ram_bytes=0, oracle_specific=False,
            cross_platform_os=(), serial_group=None, isolation_group=None,
            isolation_verified=False, qualification=False,
            classification_source=source, confidence=0.0,
            reason="Fixture capabilities are mapped, but host-platform conditions need explicit verified requirements",
            fixtures=fixtures, platform_clues=clues,
        )

    allowed_fields = {
        "operating_systems", "architectures", "features", "docker_architecture",
        "minimum_ram_bytes", "oracle_specific", "cross_platform_os", "serial_group",
        "isolation_group", "isolation_verified", "qualification", "verified",
        "fixture_dependencies_verified", "reason", "validation_safe",
    }
    if set(metadata) - allowed_fields:
        raise ClassificationError(f"unknown scheduling metadata fields for {test['nodeid']}")
    verified = metadata.get("verified", source.startswith("pytest-marker"))
    if verified is not True:
        return TestClassification(
            test_id=test["nodeid"], status="UNCLASSIFIED", categories=("UNCLASSIFIED",),
            allowed_platforms=(), required_features=(), docker_architecture=None,
            minimum_ram_bytes=0, oracle_specific=False, cross_platform_os=(),
            serial_group=None, isolation_group=None, isolation_verified=False,
            qualification=False, classification_source=source, confidence=0.0,
            reason="Metadata is not explicitly verified", fixtures=fixtures, platform_clues=clues,
        )

    allowed_platforms = _platforms(metadata)
    features = set(_string_list(metadata.get("features", []), "features", FEATURES))
    for fixture in fixtures:
        features.update(manifest["fixture_capabilities"].get(fixture, []))
    docker_arch = metadata.get("docker_architecture")
    if docker_arch is not None and docker_arch not in SUPPORTED_ARCHITECTURES:
        raise ClassificationError(f"unsupported docker_architecture on {test['nodeid']}")
    minimum_ram = metadata.get("minimum_ram_bytes", 256 * 1024 * 1024)
    if type(minimum_ram) is not int or minimum_ram < 0:
        raise ClassificationError(f"minimum_ram_bytes must be a non-negative integer on {test['nodeid']}")
    oracle = metadata.get("oracle_specific", False)
    isolation_verified = metadata.get("isolation_verified", False)
    qualification = metadata.get("qualification", False)
    if any(type(value) is not bool for value in (oracle, isolation_verified, qualification)):
        raise ClassificationError(f"boolean scheduling fields are invalid on {test['nodeid']}")
    cross_platform_os = _string_list(
        metadata.get("cross_platform_os", []), "cross_platform_os", SUPPORTED_OS,
    )
    serial_group = metadata.get("serial_group")
    isolation_group = metadata.get("isolation_group")
    if serial_group is not None and (not isinstance(serial_group, str) or not serial_group.strip()):
        raise ClassificationError(f"serial_group must be non-empty text on {test['nodeid']}")
    if isolation_group is not None and (
        not isinstance(isolation_group, str) or not isolation_group.strip()
    ):
        raise ClassificationError(f"isolation_group must be non-empty text on {test['nodeid']}")
    if isolation_group and isolation_verified:
        raise ClassificationError(
            f"isolation_group and isolation_verified are mutually exclusive on {test['nodeid']}"
        )

    conflicts = []
    if not allowed_platforms:
        conflicts.append("operating-system and architecture requirements have no supported intersection")
    if "docker" in features and docker_arch and not any(
        architecture == docker_arch for _, architecture in allowed_platforms
    ):
        conflicts.append("Docker architecture is incompatible with allowed worker architectures")
    if oracle and len(cross_platform_os) > 1:
        conflicts.append("Oracle-specific tests cannot claim multiple OS targets without explicit per-target authorization")
    if cross_platform_os and any(
        system not in {os_name for os_name, _ in allowed_platforms}
        for system in cross_platform_os
    ):
        conflicts.append("cross-platform target conflicts with allowed platform requirements")

    categories = []
    if allowed_platforms == SUPPORTED:
        categories.append("PORTABLE")
    operating_systems = {system for system, _ in allowed_platforms}
    architectures = {architecture for _, architecture in allowed_platforms}
    if operating_systems == {"Windows"}:
        categories.append("WINDOWS_ONLY")
    if operating_systems == {"Linux"}:
        categories.append("LINUX_ONLY")
    if architectures == {"arm64"}:
        categories.append("ARM64_NATIVE")
    if {"playwright", "chromium"} <= features:
        categories.append("BROWSER")
    if "docker" in features:
        categories.append("DOCKER")
    if oracle:
        categories.append("ORACLE_SPECIFIC")
    if cross_platform_os:
        categories.append("CROSS_PLATFORM")
    if serial_group is not None:
        categories.append("SERIAL_ONLY")
    return TestClassification(
        test_id=test["nodeid"], status="CONFLICT" if conflicts else "CLASSIFIED",
        categories=tuple(categories), allowed_platforms=tuple(sorted(allowed_platforms)),
        required_features=tuple(sorted(features)), docker_architecture=docker_arch,
        minimum_ram_bytes=minimum_ram, oracle_specific=oracle,
        cross_platform_os=tuple(sorted(cross_platform_os)), serial_group=serial_group,
        isolation_group=isolation_group, isolation_verified=isolation_verified,
        qualification=qualification, classification_source=source,
        confidence=1.0 if source.startswith("manifest:") else 0.85 if source == "fixture-capability-map" else 0.9,
        reason="; ".join(conflicts) if conflicts else metadata.get("reason"),
        fixtures=fixtures, platform_clues=clues,
    )


def stable_test_id(nodeid: str) -> str:
    if not isinstance(nodeid, str) or not nodeid.strip() or "\x00" in nodeid:
        raise ClassificationError("pytest nodeid must be non-empty text")
    path = PurePosixPath(nodeid.split("::", 1)[0].replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or re.match(r"[A-Za-z]:", str(path)):
        raise ClassificationError("pytest nodeid must be repository-relative")
    # Only the file component is a path. Parameter IDs are opaque pytest text
    # (including backslashes, brackets and additional '::' separators).
    filename, separator, suffix = nodeid.partition("::")
    return filename.replace("\\", "/") + separator + suffix
