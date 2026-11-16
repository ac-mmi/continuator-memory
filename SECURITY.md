# Security Policy

## Supported versions

Continuator is currently pre-1.0 (`0.1.x`). Security fixes are applied on a best-effort basis to the latest `main` / published `0.1.x` sources.

## Product security posture (current)

Continuator is designed for **local-first, single-process** use:

- The HTTP server (`ThreadingHTTPServer`) defaults to binding `127.0.0.1`.
- There is **no authentication**, authorization, or multi-tenancy.
- Do **not** expose the API to untrusted networks without placing it behind your own auth / TLS gateway.
- SQLite databases and Hugging Face model caches may contain sensitive conversation or document content — treat filesystem permissions accordingly.

## Reporting a vulnerability

Please report security issues privately to the repository maintainers (GitHub Security Advisories preferred when enabled on the repository, otherwise a private contact listed on the GitHub org/user profile).

Include:

- Affected version / commit SHA
- Reproduction steps
- Impact assessment (data exposure, RCE, denial of service, etc.)

Please allow reasonable time for a fix before public disclosure.

## Out of scope (for now)

- Multi-tenant cloud hardening
- Guarantees against malicious local users with filesystem access to the DB
