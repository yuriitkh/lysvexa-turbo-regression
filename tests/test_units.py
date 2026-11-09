"""Unit tests for classification, admission, capacity, capabilities and redaction. Synthetic inputs only."""
from __future__ import annotations

import logging

import pytest

from turbo_regression.admission import RISKY_RESOURCES, requirements_are_clean
from turbo_regression.auto_classification import classify_evidence
from turbo_regression.capabilities import SUPPORTED, detect_capabilities, normalize_architecture
from turbo_regression.classification import ClassificationError, classify_test, load_manifest, stable_test_id
from turbo_regression.execution_capacity import (
    allowed_pytest_processes, assignments_are_parallel_safe, claims_allowed, scratch_has_room, tightest_free_bytes,
)
from turbo_regression.redaction import REDACTED, RedactingFormatter, redact, redact_mapping

GIB = 1024 ** 3
COMMIT = "0123456789abcdef0123456789abcdef01234567"


# ------------------------------------------------------------------------------------------------ test IDs
def test_a_test_id_is_repository_relative_and_keeps_parameters_verbatim():
    assert stable_test_id("tests\\a.py::test_x[a\\b::c]") == "tests/a.py::test_x[a\\b::c]"


@pytest.mark.parametrize("value", ["", "   ", None, 5, "/abs/a.py::t", "../a.py::t", "tests/../../a.py::t", "C:/a.py::t", "a\x00b.py::t"])
def test_unsafe_test_ids_are_refused(value):
    with pytest.raises(ClassificationError):
        stable_test_id(value)


# ------------------------------------------------------------------------------------------------ marker / manifest classification
def record(nodeid="tests/a.py::t", **extra):
    return {"nodeid": nodeid, "fixtures": [], "markers": [], "platform_clues": [], **extra}


def test_a_test_without_any_metadata_is_unclassified():
    result = classify_test(record())
    assert result.status == "UNCLASSIFIED" and result.allowed_platforms == ()


def test_a_platform_marker_classifies_the_test():
    result = classify_test(record(markers=[{"name": "linux_only"}]))
    assert result.status == "CLASSIFIED" and {system for system, _ in result.allowed_platforms} == {"Linux"}
    assert "LINUX_ONLY" in result.categories


def test_contradicting_markers_are_a_conflict_not_a_guess():
    result = classify_test(record(markers=[{"name": "windows_only"}, {"name": "linux_only"}]))
    assert result.status == "CONFLICT" and result.reason


def test_a_skipif_marker_alone_is_not_proof_of_platform_eligibility():
    result = classify_test(record(markers=[{"name": "skipif"}]))
    assert result.status == "UNCLASSIFIED" and "skipif" in result.reason


def test_manifest_metadata_must_be_explicitly_verified():
    unverified = {"schema_version": 1, "tests": {"tests/a.py::t": {"operating_systems": ["Linux"]}}}
    assert classify_test(record(), unverified).status == "UNCLASSIFIED"
    verified = {"schema_version": 1, "tests": {"tests/a.py::t": {"operating_systems": ["Linux"], "verified": True}}}
    assert classify_test(record(), verified).status == "CLASSIFIED"


def test_a_file_level_manifest_entry_covers_its_tests():
    manifest = {"schema_version": 1, "tests": {"tests/a.py": {"operating_systems": ["Windows"], "verified": True}}}
    assert classify_test(record("tests/a.py::test_one"), manifest).status == "CLASSIFIED"


@pytest.mark.parametrize("entry", [
    {"verified": True, "bogus": 1},
    {"verified": True, "operating_systems": ["Solaris"]},
    {"verified": True, "minimum_ram_bytes": -1},
    {"verified": True, "oracle_specific": "yes"},
    {"verified": True, "isolation_group": "g", "isolation_verified": True},
    {"verified": True, "serial_group": "  "},
])
def test_invalid_manifest_metadata_is_refused(entry):
    with pytest.raises(ClassificationError):
        classify_test(record(), {"schema_version": 1, "tests": {"tests/a.py::t": entry}})


@pytest.mark.parametrize("manifest", [{"schema_version": 2}, {"schema_version": 1, "surprise": {}}, {"schema_version": 1, "tests": []}])
def test_invalid_manifests_are_refused(manifest):
    with pytest.raises(ClassificationError):
        load_manifest(manifest)


# ------------------------------------------------------------------------------------------------ evidence-based classification
def test_source_evidence_gives_at_most_an_inference_never_verification(tmp_path):
    source = tmp_path / "test_registry_access.py"
    source.write_text("import winreg\n\n\ndef test_reads_the_registry():\n    assert winreg\n", encoding="utf-8")
    analysis = classify_evidence(record(source_file=str(source)), source_commit_sha=COMMIT)
    assert analysis["confidence_level"] == "INFERRED"
    assert "WINDOWS_ONLY" in analysis["categories"] and analysis["requirements"]["operating_systems"] == ["Windows"]


def test_an_unreadable_source_is_not_silently_ignored(tmp_path):
    analysis = classify_evidence(record(source_file=str(tmp_path / "missing.py")), source_commit_sha=COMMIT)
    assert analysis["confidence_level"] != "VERIFIED"


def test_an_unresolved_fixture_blocks_verification():
    analysis = classify_evidence(record(fixtures=["some_project_fixture"]), source_commit_sha=COMMIT)
    assert analysis["confidence_level"] == "UNKNOWN" and analysis["blockers"]


def test_a_builtin_fixture_needs_no_resolution():
    analysis = classify_evidence(record(fixtures=["tmp_path"]), source_commit_sha=COMMIT)
    assert not analysis["blockers"]


# ------------------------------------------------------------------------------------------------ admission
def clean(**changes):
    value = {"status": "CLASSIFIED", "blockers": [], "isolation": [], "requirements": {}}
    value.update(changes)
    return value


def test_a_clean_analysis_is_admitted():
    assert requirements_are_clean(clean()) is True
    assert requirements_are_clean(clean(status="UNCLASSIFIED")) is True


@pytest.mark.parametrize("changes", [
    {"status": "CONFLICT"},
    {"blockers": ["unresolved fixture: x"]},
    {"isolation": [{"resource": "shared_database"}]},
    {"requirements": {"isolation_verified": False}},
    {"requirements": {"oracle_specific": True}},
    {"requirements": {"serial_group": "g"}},
    {"requirements": {"isolation_group": "g"}},
])
def test_anything_that_needs_a_lock_or_a_review_is_not_clean(changes):
    assert requirements_are_clean(clean(**changes)) is False


def test_the_risky_resource_list_is_the_documented_one():
    assert RISKY_RESOURCES == {"browser_profile", "docker_resource", "external_service", "filesystem_lock", "fixed_tcp_port",
                               "global_environment", "shared_database"}


# ------------------------------------------------------------------------------------------------ capacity
def test_one_process_is_the_default_and_parallelism_is_bounded_by_ram_and_cpu():
    base = dict(cpu_cores=16, available_ram_bytes=8 * GIB, parallel_safe=True)
    assert allowed_pytest_processes(1, **base) == 1
    assert allowed_pytest_processes(4, **base) == 4
    assert allowed_pytest_processes(4, **{**base, "parallel_safe": False}) == 1
    assert allowed_pytest_processes(4, **{**base, "available_ram_bytes": 1 * GIB}) == 1, "under memory pressure"
    assert allowed_pytest_processes(4, **{**base, "cpu_cores": 2}) == 1, "half the CPUs stay free"
    assert allowed_pytest_processes(4, **{**base, "available_ram_bytes": 3 * GIB}) == 4


@pytest.mark.parametrize("requested", [0, 5, True, 2.0, "2"])
def test_an_out_of_range_process_request_is_refused(requested):
    with pytest.raises(ValueError):
        allowed_pytest_processes(requested, cpu_cores=8, available_ram_bytes=8 * GIB, parallel_safe=True)


def test_only_independent_verified_tests_may_share_a_batch():
    assert assignments_are_parallel_safe([{"isolation_verified": True}, {"isolation_verified": True}]) is True
    assert assignments_are_parallel_safe([{"isolation_verified": False}]) is False
    assert assignments_are_parallel_safe([{"isolation_verified": True, "isolation_group": "g"}]) is False
    assert assignments_are_parallel_safe([{"isolation_verified": True, "required_capabilities": {"features": ["docker"]}}]) is False


def test_claims_stop_when_ram_disk_or_the_cgroup_is_short():
    ok = dict(available_ram_bytes=8 * GIB, available_disk_bytes=10 * GIB)
    assert claims_allowed(**ok) is True
    assert claims_allowed(**{**ok, "available_ram_bytes": 1 * GIB}) is False
    assert claims_allowed(**{**ok, "available_disk_bytes": 1}) is False
    assert claims_allowed(**ok, cgroup_available_bytes=1) is False
    assert claims_allowed(**ok, cgroup_available_bytes=4 * GIB) is True


def test_free_space_is_measured_on_the_tightest_filesystem(tmp_path):
    assert tightest_free_bytes([tmp_path]) > 0
    assert scratch_has_room([tmp_path], minimum=1) is True
    assert scratch_has_room([tmp_path], minimum=1 << 62) is False
    assert scratch_has_room([], minimum=1) is False


# ------------------------------------------------------------------------------------------------ capabilities
def test_architectures_are_normalised():
    assert {normalize_architecture(value) for value in ("AMD64", "x86_64", "x64")} == {"x64"}
    assert {normalize_architecture(value) for value in ("aarch64", "ARM64")} == {"arm64"}
    assert normalize_architecture("riscv64") == "riscv64"


def test_an_unsupported_platform_is_refused_before_any_probing():
    assert ("Windows", "x64") in SUPPORTED and ("Linux", "arm64") in SUPPORTED
    with pytest.raises(ValueError):
        detect_capabilities(system="Plan9", machine="x86_64")
    with pytest.raises(ValueError):
        detect_capabilities(system="Linux", machine="riscv64")


# ------------------------------------------------------------------------------------------------ redaction
def test_credentials_and_bearer_tokens_are_redacted_from_text():
    tail = "".join(["Ab", "Cd", "Ef", "Gh", "12", "34", "56", "78", "90"])             # built at run time: no secret-shaped literal in the source
    sample = "lysvexa-worker-" + tail
    for text in (f"enrolled with {sample}", f"Authorization: Bearer {sample}", f'{{"credential": "{sample}"}}', f"credential={sample}"):
        cleaned = redact(text)
        assert sample not in cleaned and tail not in cleaned and REDACTED in cleaned


def test_ordinary_text_is_left_alone():
    assert redact("worker windows-1 finished 12 tests") == "worker windows-1 finished 12 tests"


def test_secret_fields_are_replaced_in_nested_mappings_without_mutating_the_input():
    original = {"credential": "abcd1234", "nested": {"authorization": "xyz98765", "count": 3}, "empty": None}
    cleaned = redact_mapping(original)
    assert cleaned["credential"] == REDACTED and cleaned["nested"]["authorization"] == REDACTED and cleaned["nested"]["count"] == 3
    assert original["credential"] == "abcd1234" and original["nested"]["authorization"] == "xyz98765"


def test_the_log_formatter_redacts_the_final_line_including_tracebacks():
    tail = "".join(["Zy", "Xw", "Vu", "Ts", "09", "87", "65", "43", "21"])
    sample = "lysvexa-enroll-" + tail
    formatter = RedactingFormatter("%(message)s")
    try:
        raise RuntimeError(f"bad token {sample}")
    except RuntimeError:
        import sys
        entry = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed with %s", (sample,), sys.exc_info())
    line = formatter.format(entry)
    assert sample not in line and tail not in line
