"""Manual GPU smoke — not part of pytest.

Phase 1: load ACE-Step turbo, generate ~10s instrumental, print the path, exit 0.
"""

from __future__ import annotations


def main() -> int:
    print("GPU smoke is not implemented yet. See SPEC.md §10 and phase 1.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
