# Security

This repository is a source showcase, not a service. It contains no network listener, no credentials and no code that runs tests, so there is nothing here to deploy or expose.

## Do not

* point these modules at real controllers, workers or customer systems;
* treat the scheduling heuristics as a sandbox: a worker that executes pytest executes arbitrary Python, and nothing in this repository isolates it;
* put real credentials, enrollment tokens or host details into issues, pull requests or test fixtures. The tests use synthetic values only.

## What the code does for security

* The classifier reads test source with `ast` and never imports or runs it.
* Enrollment tokens and worker credentials are random, single-purpose and stored only as SHA-256 digests; expired, used or revoked ones are refused.
* A worker is not trusted to execute tests until that is granted explicitly, and cannot change its platform after enrollment.
* `redaction.py` removes credentials and tokens from log text, mappings and formatted tracebacks.

See *Trust assumptions and known limits* in `ARCHITECTURE.md` for what the registry design assumes.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting (Security tab, "Report a vulnerability"). Please do not open a public issue for a suspected vulnerability, and do not include real keys or infrastructure details in a report.
