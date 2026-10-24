"""Validate the configured database dialect without requiring a running database."""

import re
from pathlib import Path

from sqlalchemy import create_engine


def test_compose_database_driver_loads():
    configuration = Path("compose.yaml").read_text()
    uri = re.search(r"--backend-store-uri (\S+)", configuration).group(1)
    engine = create_engine(uri)
    try:
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.dbapi is not None
    finally:
        engine.dispose()
