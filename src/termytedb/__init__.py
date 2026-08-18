"""TermyteDB Milestone 1 public engine."""

import sys

from .api import client, schemas, service
from .api.client import TermyteDBClient, TermyteDBError
from .core import errors, logging, redaction
from .evaluation import longmemeval_extraction
from .memory import extraction, extractor, processor, provider
from .retrieval import context, embedding, retrieval
from .runtime import engine as runtime_engine
from .runtime import operations
from .runtime.engine import TermyteDB
from .storage import db, integrity, repository

_COMPAT_MODULES = {
    "client": client, "context": context, "db": db, "embedding": embedding,
    "engine": runtime_engine,
    "errors": errors, "extraction": extraction, "extractor": extractor,
    "integrity": integrity, "logging": logging, "longmemeval_extraction": longmemeval_extraction,
    "operations": operations, "processor": processor, "provider": provider,
    "redaction": redaction, "repository": repository, "retrieval": retrieval,
    "schemas": schemas, "service": service,
}
for _name, _module in _COMPAT_MODULES.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)

__all__ = ["TermyteDB", "TermyteDBClient", "TermyteDBError"]
