from __future__ import annotations

import site
import sys


def _prefer_project_runtime_packages() -> None:
    """Avoid leaking incompatible user-site packages into bundled runtimes."""
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if isinstance(user_site, str):
        user_sites = {user_site}
    else:
        user_sites = set(user_site)
    sys.path[:] = [path for path in sys.path if path not in user_sites]


_prefer_project_runtime_packages()

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
