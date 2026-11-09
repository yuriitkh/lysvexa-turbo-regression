"""Scheduler tests on synthetic data: no test is run, no network, no real host (see examples/demo.py for the fixtures)."""
from __future__ import annotations

import pytest

import demo
from turbo_regression.scheduler import SchedulingError, create_plan, locks_conflict, cross_worker_scopes, batch_is_exclusive

SCHEDULED = {demo.TEST_ADD, demo.TEST_WINDOWS, demo.TEST_SYSTEMD, demo.TEST_SERIAL}


@pytest.fixture
def fleet(tmp_path):
    return demo.build_registry(tmp_path)


def outcomes(plan):
    return {item["test_id"]: item for item in plan["test_outcomes"]}


def planned(registry, **options):
    options.setdefault("validation", demo.validation_evidence())
    return demo.make_plan(registry, **options)


def test_verified_tests_are_scheduled_and_nothing_is_executed(fleet):
    registry, _ = fleet
    plan = planned(registry)
    assert plan["status"] == "PLANNED" and plan["execution_started"] is False
    by_id = outcomes(plan)
    assert {test_id for test_id, item in by_id.items() if item["status"] == "SCHEDULED"} == SCHEDULED
    assert {batch["test_ids"][0] for batch in plan["batches"]} == SCHEDULED


def test_platform_restricted_tests_go_to_a_matching_worker(fleet):
    registry, workers = fleet
    by_id = outcomes(planned(registry))
    assert by_id[demo.TEST_WINDOWS]["assigned_worker_id"] == workers["windows"]["device_id"]
    assert by_id[demo.TEST_SYSTEMD]["assigned_worker_id"] == workers["arm"]["device_id"]


def test_a_test_without_metadata_is_refused_not_guessed(fleet):
    registry, _ = fleet
    item = outcomes(planned(registry))[demo.TEST_UNKNOWN]
    assert item["status"] == "UNCLASSIFIED" and item["reasons"]


def test_a_serial_group_becomes_an_explicit_lock(fleet):
    registry, _ = fleet
    (batch,) = [b for b in planned(registry)["batches"] if b["test_ids"] == [demo.TEST_SERIAL]]
    assert batch["execution_mode"] == "SERIAL" and "file-database" in batch["serial_locks"]


def test_without_validation_evidence_nothing_is_scheduled(fleet):
    registry, _ = fleet
    plan = planned(registry, validation=None)
    assert plan["batches"] == [] and all(item["status"] != "SCHEDULED" for item in plan["test_outcomes"])


def test_validation_recorded_for_another_commit_is_not_evidence(fleet):
    registry, _ = fleet
    plan = planned(registry, validation=demo.validation_evidence(commit="f" * 40))
    assert plan["batches"] == []


def test_a_changed_requirement_invalidates_the_recorded_validation(fleet):
    registry, _ = fleet
    evidence = demo.validation_evidence()
    evidence["validations"][demo.TEST_ADD]["requirements_sha256"] = "0" * 64
    by_id = outcomes(planned(registry, validation=evidence))
    assert by_id[demo.TEST_ADD]["status"] != "SCHEDULED"
    assert by_id[demo.TEST_WINDOWS]["status"] == "SCHEDULED"


def test_a_worker_without_execution_trust_gets_nothing(tmp_path):
    registry, _ = demo.build_registry(tmp_path, trusted=False)
    plan = planned(registry)
    item = outcomes(plan)[demo.TEST_ADD]
    assert plan["batches"] == [] and item["status"] == "BLOCKED"
    assert any("not explicitly trusted" in reason for reason in item["reasons"])


def test_a_missing_platform_is_named(tmp_path):
    registry, workers = demo.build_registry(tmp_path, names=("windows",))
    by_id = outcomes(planned(registry))
    assert by_id[demo.TEST_SYSTEMD]["status"] == "BLOCKED" and by_id[demo.TEST_SYSTEMD]["reasons"][0] == "NO_DEVICE_FOR_REQUIRED_PLATFORM"
    assert by_id[demo.TEST_WINDOWS]["assigned_worker_id"] == workers["windows"]["device_id"]


def test_an_empty_fleet_blocks_every_verified_test(tmp_path):
    registry, _ = demo.build_registry(tmp_path, names=())
    by_id = outcomes(planned(registry))
    assert by_id[demo.TEST_ADD]["status"] == "BLOCKED" and by_id[demo.TEST_ADD]["reasons"][0] == "NO_REGISTERED_DEVICES"


def test_planning_is_deterministic(fleet):
    registry, _ = fleet
    assert planned(registry) == planned(registry)


@pytest.mark.parametrize("mode", ["", "TURBO", None])
def test_unknown_modes_are_refused(fleet, mode):
    registry, _ = fleet
    with pytest.raises(SchedulingError):
        demo.make_plan(registry, mode=mode)


@pytest.mark.parametrize("commit", ["", "abc", "G" * 40, "0" * 39])
def test_the_source_commit_must_be_a_full_sha(fleet, commit):
    registry, _ = fleet
    with pytest.raises(SchedulingError):
        demo.make_plan(registry, commit=commit)


def test_focused_mode_needs_explicit_ids_and_reports_unknown_ones(fleet):
    registry, _ = fleet
    with pytest.raises(SchedulingError):
        demo.make_plan(registry, mode="FOCUSED")
    plan = create_plan(registry, demo.INVENTORY, mode="FOCUSED", source_commit_sha=demo.COMMIT, manifest=demo.MANIFEST,
                       selected_test_ids=["tests/missing.py::test_x"], planning_time=demo.NOW)
    (item,) = plan["test_outcomes"]
    assert item["status"] == "BLOCKED" and item["reasons"] == ["TEST_NOT_FOUND_IN_COLLECTION"]


def test_affected_mode_says_it_is_not_available_instead_of_pretending(fleet):
    registry, _ = fleet
    plan = demo.make_plan(registry, mode="AFFECTED")
    assert plan["status"] == "NOT_AVAILABLE" and plan["batches"] == [] and plan["execution_started"] is False


def test_duplicate_inventory_ids_are_refused(fleet):
    registry, _ = fleet
    with pytest.raises(SchedulingError):
        demo.make_plan(registry, inventory=[*demo.INVENTORY, demo.INVENTORY[0]])


def test_locks_conflict_when_one_is_the_broad_form_of_the_other():
    assert locks_conflict("resource:shared_database", "resource:shared_database")
    assert locks_conflict("resource:shared_database", "resource:shared_database:app.sqlite3")
    assert not locks_conflict("resource:shared_database:a.sqlite3", "resource:shared_database:b.sqlite3")
    assert not locks_conflict("resource:fixed_tcp_port", "resource:shared_database")


def test_only_risky_resources_that_are_not_batch_exclusive_become_cross_worker_scopes():
    isolation = [
        {"resource": "shared_database", "scope": "app.sqlite3"},
        {"resource": "fixed_tcp_port"},
        {"resource": "global_environment", "batch_exclusive": True},
        {"resource": "temp_directory"},
    ]
    assert cross_worker_scopes(isolation) == ["app.sqlite3", "fixed_tcp_port"]
    assert batch_is_exclusive(isolation) is True
    assert cross_worker_scopes("not a list") == [] and batch_is_exclusive(None) is False


def test_the_demo_runs_and_never_prints_a_credential(capsys):
    assert demo.main() == 0
    output = capsys.readouterr().out
    assert "SCHEDULED" in output and "UNCLASSIFIED" in output and "lysvexa-worker-" not in output.replace("lysvexa-worker-<redacted>", "")
