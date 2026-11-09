"""Worker registry tests on a throw-away SQLite file and a controllable clock."""
from __future__ import annotations

from datetime import timedelta

import pytest

import demo
from turbo_regression.registry import DeviceRegistry, RegistryError

WINDOWS_ID, WINDOWS_NAME, WINDOWS_CAPS = demo.WORKERS["windows"]
ARM_CAPS = demo.WORKERS["arm"][2]


class Clock:
    def __init__(self):
        self.now = demo.NOW

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry(tmp_path, clock):
    return DeviceRegistry(tmp_path / "registry.sqlite3", clock=clock)


def enroll(registry, worker_id=WINDOWS_ID, caps=WINDOWS_CAPS):
    token = registry.issue_enrollment_token(worker_id=worker_id)
    return token, registry.enroll(token, WINDOWS_NAME, caps, worker_id=worker_id)


def failure(callable_, *arguments, **keywords):
    with pytest.raises(RegistryError) as raised:
        callable_(*arguments, **keywords)
    return raised.value.status, raised.value.code


def test_an_enrolled_worker_is_online_but_not_trusted_to_execute(registry):
    _, enrolled = enroll(registry)
    device = registry.get_device(enrolled["device_id"])
    assert device["status"] == "ONLINE" and device["worker_id"] == WINDOWS_ID
    assert device["execution_authorized"] is False, "trust is a separate, explicit step"
    assert registry.set_execution_authorized(enrolled["device_id"], True)["execution_authorized"] is True


def test_an_enrollment_token_works_once(registry):
    token, _ = enroll(registry)
    assert failure(registry.enroll, token, "Another", WINDOWS_CAPS, worker_id=WINDOWS_ID) in {(401, "ENROLLMENT_INVALID"), (409, "WORKER_ID_ALREADY_ENROLLED")}
    token2 = registry.issue_enrollment_token()
    registry.enroll(token2, "Second", ARM_CAPS)
    assert failure(registry.enroll, token2, "Third", ARM_CAPS) == (401, "ENROLLMENT_INVALID")


def test_an_expired_token_is_refused(registry, clock):
    token = registry.issue_enrollment_token(ttl_seconds=10)
    clock.now += timedelta(seconds=11)
    assert failure(registry.enroll, token, "Late", WINDOWS_CAPS) == (401, "ENROLLMENT_INVALID")


def test_a_token_bound_to_a_worker_id_cannot_enroll_another(registry):
    token = registry.issue_enrollment_token(worker_id="worker-a")
    assert failure(registry.enroll, token, "Other", WINDOWS_CAPS, worker_id="worker-b") == (401, "ENROLLMENT_ID_MISMATCH")


def test_a_worker_id_cannot_be_enrolled_twice(registry):
    enroll(registry)
    second = registry.issue_enrollment_token(worker_id=WINDOWS_ID)
    assert failure(registry.enroll, second, "Again", WINDOWS_CAPS, worker_id=WINDOWS_ID) == (409, "WORKER_ID_ALREADY_ENROLLED")


@pytest.mark.parametrize("worker_id", ["../x", "", "a b", "-lead", "x" * 64])
def test_unsafe_worker_ids_are_refused(registry, worker_id):
    assert failure(registry.issue_enrollment_token, worker_id=worker_id) == (400, "INVALID_WORKER_ID")


def test_a_non_positive_ttl_is_refused(registry):
    with pytest.raises(ValueError):
        registry.issue_enrollment_token(ttl_seconds=0)


def test_neither_the_token_nor_the_credential_is_stored_in_clear(registry, tmp_path):
    token, enrolled = enroll(registry)
    data = (tmp_path / "registry.sqlite3").read_bytes()
    assert token.encode() not in data and enrolled["credential"].encode() not in data
    assert str(registry.audit_events()).count(enrolled["credential"]) == 0 and token not in str(registry.audit_events())


def test_a_heartbeat_updates_the_worker_and_a_replayed_sequence_is_stale(registry):
    _, enrolled = enroll(registry)
    credential = enrolled["credential"]
    assert registry.heartbeat(credential, WINDOWS_CAPS, sequence=1)["last_heartbeat_sequence"] == 1
    assert failure(registry.heartbeat, credential, WINDOWS_CAPS, sequence=1) == (409, "STALE_HEARTBEAT")
    assert registry.heartbeat(credential, WINDOWS_CAPS)["last_heartbeat_sequence"] == 2


def test_a_worker_cannot_change_its_platform_by_heartbeat(registry):
    _, enrolled = enroll(registry)
    assert failure(registry.heartbeat, enrolled["credential"], ARM_CAPS) == (400, "WORKER_PLATFORM_CHANGED")


def test_an_unknown_credential_is_unauthorized(registry):
    enroll(registry)
    assert failure(registry.heartbeat, "lysvexa-worker-not-a-real-credential", WINDOWS_CAPS) == (401, "WORKER_UNAUTHORIZED")
    assert failure(registry.heartbeat, "x" * 300, WINDOWS_CAPS) == (401, "WORKER_UNAUTHORIZED")


def test_a_silent_worker_goes_offline(registry, clock):
    _, enrolled = enroll(registry)
    clock.now += timedelta(seconds=registry.heartbeat_timeout + 1)
    assert registry.get_device(enrolled["device_id"])["status"] == "OFFLINE"


def test_a_revoked_worker_is_unavailable_and_its_credential_stops_working(registry):
    _, enrolled = enroll(registry)
    registry.revoke_device(enrolled["device_id"])
    assert registry.get_device(enrolled["device_id"])["status"] == "UNAVAILABLE"
    assert failure(registry.heartbeat, enrolled["credential"], WINDOWS_CAPS) == (401, "WORKER_UNAUTHORIZED")
    assert failure(registry.revoke_device, enrolled["device_id"]) == (404, "DEVICE_NOT_FOUND")


def test_the_receipt_key_is_only_given_to_a_live_credential(registry):
    _, enrolled = enroll(registry)
    key = registry.authenticated_receipt_key(enrolled["credential"])
    assert isinstance(key, bytes) and len(key) == 32
    registry.revoke_device(enrolled["device_id"])
    assert failure(registry.authenticated_receipt_key, enrolled["credential"]) == (401, "WORKER_UNAUTHORIZED")


@pytest.mark.parametrize("change", [
    lambda caps: caps.pop("cpu_cores"),
    lambda caps: caps.update(hostname="some-host"),
    lambda caps: caps.update(cpu_cores=True),
    lambda caps: caps.update(cpu_cores=0),
    lambda caps: caps.update(available_ram_bytes=-1),
    lambda caps: caps.update(pytest_available="yes"),
    lambda caps: caps.update(optional_capabilities=["unknown_thing"]),
    lambda caps: caps.update(optional_capabilities=["systemd", "systemd"]),
    lambda caps: caps.update(python_version=""),
    lambda caps: caps.update(executing_processes=9),
])
def test_invalid_capability_documents_are_refused(registry, change):
    caps = dict(WINDOWS_CAPS)
    change(caps)
    token = registry.issue_enrollment_token()
    assert failure(registry.enroll, token, "Bad", caps) == (400, "INVALID_CAPABILITIES")


def test_an_unsupported_platform_is_refused(registry):
    caps = dict(WINDOWS_CAPS, operating_system="Plan9")
    assert failure(registry.enroll, registry.issue_enrollment_token(), "Odd", caps) == (400, "UNSUPPORTED_PLATFORM")


def test_the_audit_trail_records_the_lifecycle(registry):
    _, enrolled = enroll(registry)
    registry.set_execution_authorized(enrolled["device_id"], True)
    registry.revoke_device(enrolled["device_id"])
    kinds = {event["event_type"] for event in registry.audit_events()}
    assert {"ENROLLMENT_TOKEN_ISSUED", "DEVICE_ENROLLED", "EXECUTION_AUTHORIZED", "DEVICE_REVOKED"} <= kinds
