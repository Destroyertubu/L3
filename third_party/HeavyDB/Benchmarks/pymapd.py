"""
兼容层：为旧版 benchmark 脚本提供 `pymapd` 接口。

上游 HeavyDB 的 Benchmarks 脚本历史上依赖 `pymapd`，但该包在新版本 Python 上
依赖的旧 `pyarrow` 已不再兼容。这里用 `heavydb`（pyheavydb）作为实现后端，
保持 `pymapd.connect(...)` 与 `pymapd.exceptions.*` 的最小可用 API。
"""

from __future__ import annotations

from heavyai import connect  # noqa: F401
from heavydb import exceptions  # noqa: F401

try:
    from heavyai import __version__ as __version__  # noqa: F401
except Exception:  # pragma: no cover
    __version__ = "unknown"
