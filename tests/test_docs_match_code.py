"""Documentation that drifts is worse than no documentation, because a reader
trusts it. Two tables are mechanically checkable against the code, so they are
checked rather than maintained by hand:

  - `docs/API.md`'s route table against the routes FastAPI actually serves
  - `docs/CONFIGURATION.md`'s variable table against the `LYRE_*` variables the
    code actually reads

Adding an endpoint or a setting without documenting it now fails the build.
"""

from __future__ import annotations

import re
from pathlib import Path

from server.app import app

ROOT = Path(__file__).resolve().parents[1]

# docs/API.md writes {id} where FastAPI writes the full parameter name.
PATH_PARAM_ALIASES = {"{project_id}": "{id}"}


def _served_routes() -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or not path.startswith("/api"):
            continue
        for alias, doc_form in PATH_PARAM_ALIASES.items():
            path = path.replace(alias, doc_form)
        for method in getattr(route, "methods", set()) or set():
            if method in ("HEAD", "OPTIONS"):
                continue
            routes.add((method, path))
    return routes


def _documented_routes() -> set[tuple[str, str]]:
    table = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    return {
        (match.group(1), match.group(2))
        for match in re.finditer(r"^\| `(GET|POST|PUT|PATCH|DELETE)` \| `([^`]+)` \|", table, re.M)
    }


def test_every_served_route_is_documented() -> None:
    undocumented = sorted(_served_routes() - _documented_routes())
    assert undocumented == [], "these routes exist but docs/API.md does not list them"


def test_every_documented_route_is_served() -> None:
    missing = sorted(_documented_routes() - _served_routes())
    assert missing == [], "docs/API.md lists routes the app does not serve"


def _environment_variables_read_by_code() -> set[str]:
    """Every `LYRE_*` name the application actually consults. Tests are
    excluded: they set variables, they do not define the contract."""
    found: set[str] = set()
    for path in sorted(ROOT.glob("**/*.py")):
        parts = set(path.parts)
        if parts & {".venv", "vendor", ".cache", ".tools", "tests"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found |= set(re.findall(r'os\.environ\.get\(\s*"(LYRE_[A-Z_]+)"', text))
        found |= set(re.findall(r'os\.environ\[\s*"(LYRE_[A-Z_]+)"', text))
    return found


def _documented_environment_variables() -> set[str]:
    table = (ROOT / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\| `(LYRE_[A-Z_]+)`", table, re.M))


def test_every_environment_variable_is_documented() -> None:
    undocumented = sorted(
        _environment_variables_read_by_code() - _documented_environment_variables()
    )
    assert undocumented == [], "these are read by the code but missing from docs/CONFIGURATION.md"


def test_documented_environment_variables_are_real() -> None:
    """`LYRE_PORT` is read through `server.config.lyre_port()` rather than
    inline, so it is allowed to be documented without a literal lookup."""
    stale = sorted(
        _documented_environment_variables() - _environment_variables_read_by_code() - {"LYRE_PORT"}
    )
    assert stale == [], "docs/CONFIGURATION.md documents variables nothing reads"
