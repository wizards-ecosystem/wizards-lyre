"""`server.storage`, `server.jobs`, and `worker.acestep_worker` are packages
whose `__init__` is a façade: it re-exports its modules' surface so callers
keep writing `storage.<name>` as if each were still one module.

A façade only works if it stays accurate. Two ways it rots: a name is listed in
`__all__` but no longer exists (an import error waiting for the first
`from ... import *`, and a lie to anyone reading it), or a name is re-exported
but left out of `__all__`, where ruff's per-file F401 exemption means nothing
flags it. Both are cheap to check and neither is caught by the rest of the
suite.
"""

from __future__ import annotations

import importlib

import pytest

FACADES = ["server.storage", "server.jobs", "worker.acestep_worker"]


@pytest.mark.parametrize("module_name", FACADES)
def test_every_name_in_all_actually_resolves(module_name: str) -> None:
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == [], f"{module_name}.__all__ names things it does not export"


@pytest.mark.parametrize("module_name", FACADES)
def test_all_has_no_duplicates(module_name: str) -> None:
    module = importlib.import_module(module_name)
    duplicates = sorted({n for n in module.__all__ if module.__all__.count(n) > 1})
    assert duplicates == []


@pytest.mark.parametrize("module_name", FACADES)
def test_every_re_export_is_listed_in_all(module_name: str) -> None:
    """Anything the façade imports from its own submodules is part of the
    surface whether or not it is declared, so it belongs in `__all__`. The
    submodules themselves are excluded -- `storage.jsonio` is reached as a
    module by tests that patch it, not re-exported as a name.
    """
    module = importlib.import_module(module_name)
    declared = set(module.__all__)

    undeclared = []
    for name, value in vars(module).items():
        if name.startswith("__"):
            continue
        origin = getattr(value, "__module__", None)
        if origin is None or not origin.startswith(module_name + "."):
            continue  # a submodule, or something imported from elsewhere
        if name not in declared:
            undeclared.append(name)

    assert sorted(undeclared) == [], (
        f"{module_name} re-exports these without listing them in __all__: {sorted(undeclared)}"
    )
