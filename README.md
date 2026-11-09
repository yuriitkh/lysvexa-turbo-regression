# Turbo Regression: test classification and scheduling

Turbo Regression decides **which pytest tests may run where, and in what order**, before anything is executed. This repository is a source showcase of its planning core: a conservative test classifier, a deterministic dry-run scheduler and a worker registry. It uses only the Python standard library.

It is **not** a runner. Nothing in this repository starts pytest, opens a network connection or talks to a machine; the planner produces a plan and says why every test that cannot run is not in it.

## What the code demonstrates

| Idea | Where |
|---|---|
| **Never guess a test's requirements.** A test is `VERIFIED` only with explicit metadata *and* recorded validation evidence for every platform it claims, at the exact commit and for the exact requirements; source analysis can only ever give `INFERRED`. | `classification.py`, `auto_classification.py` |
| **Deterministic, explainable plans.** The same inventory, registry state and clock give the same plan; every test that is not scheduled carries a reason. | `scheduler.py` |
| **Shared resources become locks.** Serial groups and risky resources (ports, databases, browser profiles) are serialised across workers; broad locks conflict with narrower ones. | `scheduler.py`, `admission.py` |
| **Workers are enrolled, not assumed.** One-time, expiring enrollment tokens; credentials stored only as hashes; heartbeat sequence numbers; a worker is not trusted to run tests until that is granted explicitly. | `registry.py` |
| **Safe by default capacity.** One pytest process unless RAM, CPU and test isolation all allow more; claims stop under memory, disk or cgroup pressure. | `execution_capacity.py`, `capabilities.py` |
| **Secrets stay out of logs.** Credentials and tokens are redacted from text, mappings and formatted tracebacks. | `redaction.py` |

## Quick start

```bash
python -m pip install -e ".[test]"
python -m pytest              # synthetic tests, a few seconds, no network
python examples/demo.py       # plans a five-test suite for two synthetic workers
```

Python 3.11 or newer. The continuous-integration workflow (`.github/workflows/ci.yml`) runs the tests and the example on Linux and Windows.

The example enrolls two synthetic workers, classifies five made-up tests, plans them with and without validation evidence and shows that a test without metadata is refused instead of guessed.

## Layout

```text
turbo_regression/
  classification.py        markers and manifests -> a conservative classification
  auto_classification.py   source analysis, fixture closure, evidence levels (VERIFIED / INFERRED / UNKNOWN)
  scheduler.py             dry-run planner: placement, locks, batches, reasons
  admission.py             risky-resource list and the "clean requirements" rule
  registry.py              SQLite worker registry: enrollment, heartbeat, trust, audit
  execution_capacity.py    how many pytest processes, and when to stop claiming
  capabilities.py          what this host can do
  redaction.py             log redaction
examples/demo.py           end-to-end planning on synthetic data
tests/                     synthetic tests for every module
ARCHITECTURE.md  SECURITY.md
```

## Provenance and status

This is real implementation code from a larger private system (controller, workers, remote execution and signed result receipts are not included). It was adapted for this repository: the package was renamed, internal comments were removed, and two small pieces that the scheduler needs from omitted modules (`RISKY_RESOURCES` and the clean-requirements rule) were extracted unchanged in behaviour into `admission.py`. The tests and the example in this repository are new.

What this repository does **not** claim: that it is a runnable distributed system, that it has been used by third parties, any performance figures, or that it can safely run untrusted tests (see `SECURITY.md`).

## Rights

No licence has been applied. Unless the owner grants permission in writing, all rights are reserved: the source is published to be read and evaluated, not to be copied, redistributed or used in production.
