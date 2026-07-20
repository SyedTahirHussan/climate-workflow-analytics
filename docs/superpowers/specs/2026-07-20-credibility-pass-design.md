# Credibility pass — design

**Date:** 2026-07-20
**Goal:** Make the repository's engineering discipline visible to PhD reviewers
at a glance, without adding maintenance surface.

## Scope

1. **GitHub Actions CI** (`.github/workflows/ci.yml`)
   - `lint` job: `ruff check src tests examples` on Python 3.12.
   - `test` job: `pytest --cov=cwa` on a Python 3.10 / 3.11 / 3.12 × Ubuntu
     matrix — verifies the `requires-python = ">=3.10"` claim on every push.
   - Coverage is printed in the job log; no external coverage service.
2. **CI badge** at the top of `README.md`.
3. **`CITATION.cff`** — citation metadata; GitHub renders a
   "Cite this repository" button.
4. **GitHub repo metadata** — About description and topics via `gh repo edit`.

## Explicitly out of scope (considered, rejected)

- **mypy CI** — the code is not annotated throughout; a half-strict job is a
  worse signal than none.
- **pre-commit hooks** — solo repository; CI already gates.
- **Dependabot** — update noise on a deliberately frozen prototype.
- **Codecov** — external account and token for marginal gain over the in-log
  coverage table.

## Verification

- `ruff check` and the full test suite pass locally before pushing
  (baseline confirmed: 0 lint findings, 24/24 tests, 50% total coverage).
- After push: the Actions run for all four jobs is green and the README badge
  renders.
