from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from audre import schema

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str, table: str) -> pd.DataFrame:
    return schema.coerce(pd.read_csv(FIXTURE_DIR / name), schema.SPECS[table])


@pytest.fixture(scope="session")
def conversations() -> pd.DataFrame:
    return _load("conversations.csv", "conversations")


@pytest.fixture(scope="session")
def labels() -> pd.DataFrame:
    return _load("labels.csv", "labels")


@pytest.fixture(scope="session")
def resource_records() -> pd.DataFrame:
    return _load("resources.csv", "resources")


@pytest.fixture(scope="session")
def documents() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "documents.csv")
