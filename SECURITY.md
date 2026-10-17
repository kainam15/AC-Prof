# Security policy

## Supported versions

Security fixes are prioritized for the current `main` branch and the next release. Published GitHub Release assets are immutable; older releases may not receive backports. Check release notes and verify downloaded files against their matching `SHA256SUMS`.

## Report a vulnerability

**Do not disclose exploit steps, credentials, or private model data in a public issue.**

Use GitHub's **Security → Report a vulnerability** for this repository if private vulnerability reporting is enabled. If that option is unavailable, open a public issue requesting a confidential reporting channel **without including vulnerability details**. A maintainer can then coordinate private contact.

Include the affected commit or release, environment, impact, and a minimal reproduction only in the confidential report. Allow time for triage before public disclosure; no specific response-time or embargo guarantee is currently offered.

## Security model

AC-Prof may execute third-party model code inside Docker, including models whose registered policy permits `trust_remote_code`. Model repositories and mirrors are independent trust boundaries. SHA256 or revision pinning improves artifact integrity but does **not** establish the safety of executable model code. Only run models from sources you trust, preferably on an isolated host without sensitive credentials.

The `acprof doctor` command reports host capabilities; it does not certify that arbitrary model code or external download sources are safe. See [installation](docs/getting_started/installation.md) and [host configuration](docs/usage/configuration.md) for prerequisites and mirror behavior.
