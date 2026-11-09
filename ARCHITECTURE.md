# Architecture

## Planning pipeline

```text
pytest inventory (records)         explicit manifest          validation evidence
          |                              |                          |
          +--------------+---------------+--------------------------+
                         v
              classification  ->  confidence level per test
                         |        VERIFIED | INFERRED | UNKNOWN
                         v
        scheduler.create_plan(registry, inventory, mode, commit)
                         |
        +----------------+-----------------+
        v                                  v
   batches (one test, one worker,    unscheduled tests, each with a reason
   locks, dependencies, estimates)   (UNCLASSIFIED, BLOCKED, NOT_APPLICABLE ...)
```

The scheduler is a **dry run**: it returns a plan and `execution_started: false`. Executing a plan is outside this repository.

## Classification and evidence

* `classification.py` turns pytest markers or an explicit manifest into a `TestClassification` (allowed platforms, features, memory, locks). Anything not explicitly verified stays `UNCLASSIFIED`; contradicting declarations are a `CONFLICT`, not a merge.
* `auto_classification.py` reads the test source and its fixture closure. It can find platform APIs, browser or Docker use and shared resources, and records them as evidence. That is **at most `INFERRED`**.
* A test becomes `VERIFIED` only when its explicit metadata is verified **and** the validation record contains a `PASS` for every platform it claims, for the plan's commit, bound to the digest of the test's current requirements. A changed requirement or a different commit silently invalidates the record.
* In `FULL` mode only `VERIFIED` tests are scheduled. `FULL_FAST` may admit an unverified test provisionally when a safe lane is confirmed (`admission.requirements_are_clean`); such tests are labelled `PROVISIONAL` and are never reported as verified.

## Scheduling

* Targets: a plan may request operating systems / architectures; a test that does not apply is `NOT_APPLICABLE`, one with no matching worker is `BLOCKED` with the reason (`NO_REGISTERED_DEVICES`, `NO_DEVICE_FOR_REQUIRED_PLATFORM`, `REQUIRED_CAPABILITIES_OR_RESOURCES_UNAVAILABLE`).
* A worker is a candidate only if it is online, **explicitly trusted for execution**, has the platform, features and RAM, and has capacity.
* Locks: a declared serial group or a shared resource becomes a `SERIAL` job with a lock; `resource:x` conflicts with `resource:x:y`. Independent verified tests are `PARALLEL`. Jobs are ordered longest first with stable tie-breaks; the result is deterministic for a given registry snapshot and clock.
* A soft preference gives Linux/ARM64 workers a head start on portable work; it never overrides a materially earlier finish elsewhere.

## Worker registry

SQLite, one connection per operation. Enrollment tokens are single-use and expire; credentials are random and only their SHA-256 is stored; heartbeats carry a strictly increasing sequence number; a worker cannot change its platform; revocation takes effect for the credential immediately; the audit trail records the lifecycle without secrets. A capability document is validated field by field and cannot carry extra data such as host names.

## Capacity

`execution_capacity.py` allows one pytest process unless the request, the free RAM, half of the CPUs and the test isolation all permit more (hard cap 4). `claims_allowed` stops claiming under RAM, disk or cgroup pressure; CPU load alone is not a reason.

## Trust assumptions and known limits

* The registry host is trusted. `authenticated_receipt_key` derives a MAC key from the stored credential digest, so whoever can read the registry database can derive it; the full system keeps the registry on the controller host. A hardened deployment would use a separate server-side secret.
* The plan names a source-preparation component (`source_preparation.preparer`) that is not part of this repository.
* The classifier reads source with the standard `ast` module; it never imports or executes the code under analysis.
* Not included: controller, worker loop, remote checkout, test execution, signed receipts, updates, user interface.
