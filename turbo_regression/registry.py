"""Persistent, execution-free registry for Turbo Regression workers."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Iterator
from uuid import uuid4

from turbo_regression.capabilities import SUPPORTED, normalize_architecture

EVENT_TYPES = frozenset({"STARTED", "COMPLETED", "FAILED"})
OPTIONAL_CAPABILITIES = frozenset({"windows_api", "powershell", "systemd", "arm64", "native_docker"})
TASK_ID = re.compile(r"[A-Za-z0-9._:-]{1,120}\Z")
WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}\Z")
CAPABILITY_FLAGS = (
    "pytest_available", "playwright_available", "chromium_available", "docker_available",
)


class RegistryError(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code
        super().__init__(code)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_stamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def validate_capabilities(value: object) -> dict:
    if not isinstance(value, dict):
        raise RegistryError(400, "INVALID_CAPABILITIES")
    required = {
        "operating_system", "architecture", "cpu_cores", "available_ram_bytes",
        "available_disk_bytes", "python_version", *CAPABILITY_FLAGS,
        "docker_architecture", "optional_capabilities",
    }
    executing = value.get("executing_processes")
    if "executing_processes" in value and (type(executing) is not int or not 0 <= executing <= 4):
        raise RegistryError(400, "INVALID_CAPABILITIES")
    if set(value) - {"executing_processes"} != required:
        raise RegistryError(400, "INVALID_CAPABILITIES")
    system, architecture = value["operating_system"], value["architecture"]
    if not isinstance(system, str) or not isinstance(architecture, str):
        raise RegistryError(400, "INVALID_CAPABILITIES")
    architecture = normalize_architecture(architecture)
    if (system, architecture) not in SUPPORTED:
        raise RegistryError(400, "UNSUPPORTED_PLATFORM")
    for key in ("cpu_cores", "available_ram_bytes", "available_disk_bytes"):
        if type(value[key]) is not int or value[key] < (1 if key == "cpu_cores" else 0):
            raise RegistryError(400, "INVALID_CAPABILITIES")
    if not isinstance(value["python_version"], str) or not value["python_version"] or len(value["python_version"]) > 32:
        raise RegistryError(400, "INVALID_CAPABILITIES")
    if any(type(value[key]) is not bool for key in CAPABILITY_FLAGS):
        raise RegistryError(400, "INVALID_CAPABILITIES")
    docker_arch = value["docker_architecture"]
    if docker_arch is not None and (not isinstance(docker_arch, str) or len(docker_arch) > 32):
        raise RegistryError(400, "INVALID_CAPABILITIES")
    optional = value["optional_capabilities"]
    if (not isinstance(optional, list) or any(not isinstance(item, str) for item in optional)
            or len(optional) != len(set(optional)) or not set(optional) <= OPTIONAL_CAPABILITIES):
        raise RegistryError(400, "INVALID_CAPABILITIES")
    return {**value, "architecture": architecture, "optional_capabilities": sorted(optional)}


class DeviceRegistry:
    def __init__(self, database: str | Path, *, heartbeat_timeout: int = 90,
                 clock=None) -> None:
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive")
        self.database = str(database)
        self.heartbeat_timeout = heartbeat_timeout
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if self.database != ":memory:":
            Path(self.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS turbo_enrollment_tokens (
                    token_hash TEXT PRIMARY KEY,
                    worker_id TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    revoked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS turbo_devices (
                    device_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    friendly_name TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    last_heartbeat_at TEXT NOT NULL,
                    last_heartbeat_sequence INTEGER NOT NULL DEFAULT 0,
                    execution_authorized INTEGER NOT NULL DEFAULT 0,
                    assigned_task TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS turbo_worker_audit (
                    audit_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    device_id TEXT,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turbo_device_credentials (
                    credential_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL REFERENCES turbo_devices(device_id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS turbo_credentials_by_hash
                    ON turbo_device_credentials(token_hash);
                CREATE TABLE IF NOT EXISTS turbo_worker_events (
                    event_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL REFERENCES turbo_devices(device_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT
                );
                CREATE INDEX IF NOT EXISTS turbo_events_pending
                    ON turbo_worker_events(device_id, acknowledged_at, created_at);
            """)
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(turbo_devices)")
            }
            if "worker_id" not in columns:
                db.execute("ALTER TABLE turbo_devices ADD COLUMN worker_id TEXT")
            if "last_heartbeat_sequence" not in columns:
                db.execute(
                    "ALTER TABLE turbo_devices ADD COLUMN last_heartbeat_sequence INTEGER NOT NULL DEFAULT 0"
                )
            if "execution_authorized" not in columns:
                db.execute(
                    "ALTER TABLE turbo_devices ADD COLUMN execution_authorized INTEGER NOT NULL DEFAULT 0"
                )
            token_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(turbo_enrollment_tokens)")
            }
            if "worker_id" not in token_columns:
                db.execute("ALTER TABLE turbo_enrollment_tokens ADD COLUMN worker_id TEXT")
            db.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS turbo_active_worker_id
                ON turbo_devices(worker_id) WHERE revoked_at IS NULL
            """)

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("registry clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def issue_enrollment_token(
        self,
        *,
        ttl_seconds: int = 3600,
        worker_id: str | None = None,
    ) -> str:
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        if worker_id is not None and (
            not isinstance(worker_id, str) or not WORKER_ID.fullmatch(worker_id)
        ):
            raise RegistryError(400, "INVALID_WORKER_ID")
        token = "lysvexa-enroll-" + secrets.token_urlsafe(32)
        now = self._now()
        with self._connection() as db:
            db.execute(
                """INSERT INTO turbo_enrollment_tokens
                   (token_hash,worker_id,created_at,expires_at) VALUES(?,?,?,?)""",
                (
                    self._hash(token), worker_id, _stamp(now),
                    _stamp(now + timedelta(seconds=ttl_seconds)),
                ),
            )
            self._audit(
                db, "ENROLLMENT_TOKEN_ISSUED",
                worker_id=worker_id,
                details={"expires_at": _stamp(now + timedelta(seconds=ttl_seconds))},
                created_at=_stamp(now),
            )
        return token

    @staticmethod
    def _audit(
        db: sqlite3.Connection,
        event_type: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
        details: dict | None = None,
        created_at: str | None = None,
    ) -> None:
        db.execute(
            """INSERT INTO turbo_worker_audit
               (audit_id,worker_id,device_id,event_type,created_at,details_json)
               VALUES(?,?,?,?,?,?)""",
            (
                str(uuid4()), worker_id, device_id, event_type,
                created_at or _stamp(datetime.now(timezone.utc)),
                json.dumps(details or {}, sort_keys=True),
            ),
        )

    def enroll(
        self,
        token: str,
        friendly_name: str,
        capabilities: object,
        *,
        worker_id: str | None = None,
    ) -> dict:
        if not isinstance(token, str) or len(token) > 200:
            raise RegistryError(401, "ENROLLMENT_INVALID")
        if not isinstance(friendly_name, str) or not friendly_name.strip() or len(friendly_name.strip()) > 120:
            raise RegistryError(400, "INVALID_DEVICE_NAME")
        clean_capabilities = validate_capabilities(capabilities)
        now = self._now()
        device_id, credential = str(uuid4()), "lysvexa-worker-" + secrets.token_urlsafe(32)
        if worker_id is None:
            worker_id = "worker-" + device_id[:8]
        if not isinstance(worker_id, str) or not WORKER_ID.fullmatch(worker_id):
            raise RegistryError(400, "INVALID_WORKER_ID")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT 1 FROM turbo_devices WHERE worker_id=? AND revoked_at IS NULL",
                (worker_id,),
            ).fetchone()
            if existing is not None:
                raise RegistryError(409, "WORKER_ID_ALREADY_ENROLLED")
            row = db.execute(
                """SELECT token_hash,worker_id FROM turbo_enrollment_tokens
                   WHERE token_hash=? AND used_at IS NULL AND revoked_at IS NULL AND expires_at>?""",
                (self._hash(token), _stamp(now)),
            ).fetchone()
            if row is None or not hmac.compare_digest(row["token_hash"], self._hash(token)):
                raise RegistryError(401, "ENROLLMENT_INVALID")
            if row["worker_id"] is not None and row["worker_id"] != worker_id:
                raise RegistryError(401, "ENROLLMENT_ID_MISMATCH")
            db.execute(
                "UPDATE turbo_enrollment_tokens SET used_at=? WHERE token_hash=? AND used_at IS NULL",
                (_stamp(now), row["token_hash"]),
            )
            db.execute(
                """INSERT INTO turbo_devices
                   (device_id,worker_id,friendly_name,capabilities_json,last_heartbeat_at,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (device_id, worker_id, friendly_name.strip(), json.dumps(clean_capabilities, sort_keys=True),
                 _stamp(now), _stamp(now), _stamp(now)),
            )
            db.execute(
                """INSERT INTO turbo_device_credentials
                   (credential_id,device_id,token_hash,created_at) VALUES(?,?,?,?)""",
                (str(uuid4()), device_id, self._hash(credential), _stamp(now)),
            )
            self._audit(
                db, "DEVICE_ENROLLED", worker_id=worker_id, device_id=device_id,
                details={
                    "operating_system": clean_capabilities["operating_system"],
                    "architecture": clean_capabilities["architecture"],
                },
                created_at=_stamp(now),
            )
        return {
            "device_id": device_id, "worker_id": worker_id,
            "credential": credential, "device": self.get_device(device_id),
        }

    def _credential_device(self, credential: str) -> str:
        if not isinstance(credential, str) or len(credential) > 200:
            raise RegistryError(401, "WORKER_UNAUTHORIZED")
        digest = self._hash(credential)
        with self._connection() as db:
            row = db.execute(
                """SELECT c.token_hash,c.device_id,d.revoked_at
                   FROM turbo_device_credentials c JOIN turbo_devices d USING(device_id)
                   WHERE c.token_hash=? AND c.revoked_at IS NULL""",
                (digest,),
            ).fetchone()
        if row is None or row["revoked_at"] is not None or not hmac.compare_digest(row["token_hash"], digest):
            raise RegistryError(401, "WORKER_UNAUTHORIZED")
        return row["device_id"]

    def authenticated_receipt_key(self, credential: str) -> bytes:
        """Return a receipt MAC key only after validating the live worker credential."""
        self._credential_device(credential)
        return bytes.fromhex(self._hash(credential))

    def heartbeat(
        self,
        credential: str,
        capabilities: object,
        sequence: int | None = None,
    ) -> dict:
        device_id = self._credential_device(credential)
        clean_capabilities = validate_capabilities(capabilities)
        now_dt = self._now()
        now = _stamp(now_dt)
        rejected_code = None
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT worker_id,capabilities_json,last_heartbeat_sequence FROM turbo_devices "
                "WHERE device_id=? AND revoked_at IS NULL",
                (device_id,),
            ).fetchone()
            if row is None:
                raise RegistryError(401, "WORKER_UNAUTHORIZED")
            if sequence is None:
                sequence = row["last_heartbeat_sequence"] + 1
            if type(sequence) is not int:
                rejected_code = "INVALID_HEARTBEAT_SEQUENCE"
                self._audit(
                    db, "HEARTBEAT_REJECTED", worker_id=row["worker_id"], device_id=device_id,
                    details={"reason": rejected_code},
                    created_at=now,
                )
            elif sequence <= row["last_heartbeat_sequence"]:
                rejected_code = "STALE_HEARTBEAT"
                self._audit(
                    db, "HEARTBEAT_REJECTED", worker_id=row["worker_id"], device_id=device_id,
                    details={"reason": rejected_code, "sequence": sequence},
                    created_at=now,
                )
            else:
                prior_capabilities = json.loads(row["capabilities_json"])
                if (
                    clean_capabilities["operating_system"] != prior_capabilities["operating_system"]
                    or clean_capabilities["architecture"] != prior_capabilities["architecture"]
                ):
                    rejected_code = "WORKER_PLATFORM_CHANGED"
                    self._audit(
                        db, "HEARTBEAT_REJECTED", worker_id=row["worker_id"], device_id=device_id,
                        details={"reason": rejected_code, "sequence": sequence},
                        created_at=now,
                    )
                else:
                    changed = sorted(
                        key for key in clean_capabilities
                        if clean_capabilities[key] != prior_capabilities.get(key)
                    )
                    db.execute(
                        """UPDATE turbo_devices SET capabilities_json=?,last_heartbeat_at=?,
                           last_heartbeat_sequence=?,updated_at=? WHERE device_id=? AND revoked_at IS NULL""",
                        (
                            json.dumps(clean_capabilities, sort_keys=True),
                            now, sequence, now, device_id,
                        ),
                    )
                    self._audit(
                        db, "HEARTBEAT_ACCEPTED", worker_id=row["worker_id"], device_id=device_id,
                        details={"sequence": sequence, "changed_capabilities": changed},
                        created_at=now,
                    )
        if rejected_code:
            status = (
                409 if rejected_code == "STALE_HEARTBEAT"
                else 400
            )
            raise RegistryError(status, rejected_code)
        return self.get_device(device_id)

    def _public_device(self, row: sqlite3.Row, now: datetime) -> dict:
        if row["revoked_at"] is not None:
            status = "UNAVAILABLE"
        elif (now - _parse_stamp(row["last_heartbeat_at"])).total_seconds() > self.heartbeat_timeout:
            status = "OFFLINE"
        elif row["assigned_task"] is not None:
            status = "BUSY"
        else:
            status = "ONLINE"
        return {
            "device_id": row["device_id"],
            "worker_id": row["worker_id"] or row["device_id"],
            "friendly_name": row["friendly_name"],
            "capabilities": json.loads(row["capabilities_json"]),
            "status": status,
            "last_heartbeat_at": row["last_heartbeat_at"],
            "last_heartbeat_sequence": row["last_heartbeat_sequence"],
            "execution_authorized": bool(row["execution_authorized"]),
            "assigned_task": row["assigned_task"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_device(self, device_id: str) -> dict:
        with self._connection() as db:
            row = db.execute("SELECT * FROM turbo_devices WHERE device_id=?", (device_id,)).fetchone()
        if row is None:
            raise RegistryError(404, "DEVICE_NOT_FOUND")
        return self._public_device(row, self._now())

    def list_devices(self) -> list[dict]:
        with self._connection() as db:
            rows = db.execute("SELECT * FROM turbo_devices ORDER BY created_at,device_id").fetchall()
        now = self._now()
        return [self._public_device(row, now) for row in rows]

    def revoke_device(self, device_id: str) -> None:
        now = _stamp(self._now())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT worker_id FROM turbo_devices WHERE device_id=? AND revoked_at IS NULL",
                (device_id,),
            ).fetchone()
            if row is None:
                raise RegistryError(404, "DEVICE_NOT_FOUND")
            result = db.execute(
                "UPDATE turbo_devices SET revoked_at=?,updated_at=? WHERE device_id=? AND revoked_at IS NULL",
                (now, now, device_id),
            )
            if result.rowcount != 1:
                raise RegistryError(404, "DEVICE_NOT_FOUND")
            db.execute(
                "UPDATE turbo_device_credentials SET revoked_at=? WHERE device_id=? AND revoked_at IS NULL",
                (now, device_id),
            )
            self._audit(
                db, "DEVICE_REVOKED", worker_id=row["worker_id"], device_id=device_id,
                created_at=now,
            )

    def set_execution_authorized(self, device_id: str, authorized: bool) -> dict:
        if type(authorized) is not bool:
            raise RegistryError(400, "INVALID_EXECUTION_AUTHORIZATION")
        now = _stamp(self._now())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT worker_id,execution_authorized FROM turbo_devices "
                "WHERE device_id=? AND revoked_at IS NULL",
                (device_id,),
            ).fetchone()
            if row is None:
                raise RegistryError(404, "DEVICE_NOT_FOUND")
            if bool(row["execution_authorized"]) != authorized:
                db.execute(
                    "UPDATE turbo_devices SET execution_authorized=?,updated_at=? WHERE device_id=?",
                    (int(authorized), now, device_id),
                )
                self._audit(
                    db,
                    "EXECUTION_AUTHORIZED" if authorized else "EXECUTION_REVOKED",
                    worker_id=row["worker_id"],
                    device_id=device_id,
                    created_at=now,
                )
        return self.get_device(device_id)

    def audit_events(self, *, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("audit limit must be between 1 and 1000")
        with self._connection() as db:
            rows = db.execute(
                """SELECT worker_id,device_id,event_type,created_at,details_json
                   FROM turbo_worker_audit ORDER BY created_at DESC,audit_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "worker_id": row["worker_id"],
                "device_id": row["device_id"],
                "event_type": row["event_type"],
                "created_at": row["created_at"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    def remove_device(self, device_id: str) -> None:
        with self._connection() as db:
            result = db.execute("DELETE FROM turbo_devices WHERE device_id=?", (device_id,))
        if result.rowcount != 1:
            raise RegistryError(404, "DEVICE_NOT_FOUND")

    def enqueue_notification(self, device_id: str, event_type: str, task_id: str) -> dict:
        if not isinstance(event_type, str) or event_type not in EVENT_TYPES \
                or not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise RegistryError(400, "INVALID_EVENT")
        now = _stamp(self._now())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM turbo_devices WHERE device_id=?", (device_id,)).fetchone()
            if row is None:
                raise RegistryError(404, "DEVICE_NOT_FOUND")
            device = self._public_device(row, self._now())
            if device["status"] not in {"ONLINE", "BUSY"}:
                raise RegistryError(409, "DEVICE_UNAVAILABLE")
            if event_type == "STARTED":
                if row["assigned_task"] is not None:
                    raise RegistryError(409, "DEVICE_BUSY")
                db.execute(
                    "UPDATE turbo_devices SET assigned_task=?,updated_at=? WHERE device_id=?",
                    (task_id, now, device_id),
                )
            else:
                if row["assigned_task"] != task_id:
                    raise RegistryError(409, "TASK_ASSIGNMENT_MISMATCH")
                db.execute(
                    "UPDATE turbo_devices SET assigned_task=NULL,updated_at=? WHERE device_id=?",
                    (now, device_id),
                )
            event = {"event_id": str(uuid4()), "device_id": device_id,
                     "event_type": event_type, "task_id": task_id, "created_at": now}
            db.execute(
                """INSERT INTO turbo_worker_events(event_id,device_id,event_type,task_id,created_at)
                   VALUES(?,?,?,?,?)""",
                (event["event_id"], device_id, event_type, task_id, now),
            )
        return event

    def pending_events(self, credential: str) -> list[dict]:
        device_id = self._credential_device(credential)
        with self._connection() as db:
            rows = db.execute(
                """SELECT event_id,device_id,event_type,task_id,created_at
                   FROM turbo_worker_events
                   WHERE device_id=? AND acknowledged_at IS NULL ORDER BY created_at,event_id LIMIT 100""",
                (device_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def acknowledge_event(self, credential: str, event_id: str) -> None:
        device_id = self._credential_device(credential)
        with self._connection() as db:
            result = db.execute(
                """UPDATE turbo_worker_events SET acknowledged_at=?
                   WHERE event_id=? AND device_id=? AND acknowledged_at IS NULL""",
                (_stamp(self._now()), event_id, device_id),
            )
        if result.rowcount != 1:
            raise RegistryError(404, "EVENT_NOT_FOUND")
