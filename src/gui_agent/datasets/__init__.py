"""Dataset adapters for the public GUI task datasets.

ScreenAgent, Mind2Web and WebArena each ship a different shape. Every adapter
translates one of them into :class:`GUITaskSample` so that validation, export and
the planner only ever see the project's own format.
"""

from .base import DatasetAdapter, DatasetError, read_records
from .mind2web import Mind2WebAdapter
from .schemas import GUIActionStep, GUITaskSample
from .screenagent import ScreenAgentAdapter
from .webarena import WebArenaAdapter

#: Adapters by the name used in ``--dataset``.
ADAPTERS: dict[str, type[DatasetAdapter]] = {
    ScreenAgentAdapter.name: ScreenAgentAdapter,
    Mind2WebAdapter.name: Mind2WebAdapter,
    WebArenaAdapter.name: WebArenaAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate an adapter by its CLI name."""
    try:
        return ADAPTERS[name]()
    except KeyError:
        known = ", ".join(sorted(ADAPTERS))
        raise DatasetError(f"unknown dataset {name!r}; known: {known}") from None


__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "DatasetError",
    "GUIActionStep",
    "GUITaskSample",
    "Mind2WebAdapter",
    "ScreenAgentAdapter",
    "WebArenaAdapter",
    "get_adapter",
    "read_records",
]
