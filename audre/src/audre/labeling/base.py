"""Labeler interface and taxonomy loading.

AUDRE separates *how* labels are produced from *what is done with them*. Any
object implementing :class:`Labeler` can feed the analysis layers, which lets a
reader reproduce the pipeline with the offline rule labeler while an
organization runs the LLM labeler internally.

Every label carries a ``label_source`` recording the labeler and, for LLM
labelers, the model that produced it. This is not bookkeeping for its own sake:
labels are *assigned*, not recorded, and downstream figures must be able to
state which system assigned them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

TAXONOMY_DIR = Path(__file__).resolve().parent.parent / "taxonomies"
DEFAULT_TAXONOMY = "default.yaml"


@dataclass
class Taxonomy:
    """A closed set of allowed values per single-valued label kind.

    Closed sets matter for two reasons. They make assignments mutually exclusive
    and therefore countable as shares, and they stop a generative model from
    inventing categories that cannot be compared across runs.
    """

    name: str
    description: str
    fields: dict[str, list[str]] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TAXONOMY) -> "Taxonomy":
        """Load a taxonomy from a YAML file, or by name from the bundled set."""

        candidate = Path(path)
        if not candidate.exists():
            candidate = TAXONOMY_DIR / str(path)
        if not candidate.exists():
            available = sorted(p.name for p in TAXONOMY_DIR.glob("*.yaml"))
            raise FileNotFoundError(f"Taxonomy {path!r} not found. Bundled: {available}")

        payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        return cls(
            name=payload.get("name", candidate.stem),
            description=payload.get("description", ""),
            fields={key: list(values) for key, values in payload.get("fields", {}).items()},
            source=payload.get("source"),
        )

    def allowed(self, label_kind: str) -> list[str]:
        if label_kind not in self.fields:
            raise KeyError(
                f"Taxonomy {self.name!r} has no field {label_kind!r}; "
                f"available: {sorted(self.fields)}"
            )
        return self.fields[label_kind]

    def is_valid(self, label_kind: str, value: str) -> bool:
        return value in set(self.allowed(label_kind))

    def coverage(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Report which taxonomy values were used, and which never appeared.

        Unused categories are informative. A taxonomy where most values never
        occur may be a poor fit for the setting, and reporting that is more
        honest than presenting only the categories that happened to be assigned.
        """

        rows = []
        for label_kind, allowed in self.fields.items():
            assigned = labels.loc[labels["label_kind"] == label_kind, "label_value"]
            counts = assigned.value_counts()
            for value in allowed:
                rows.append(
                    {
                        "label_kind": label_kind,
                        "label_value": value,
                        "n_conversations": int(counts.get(value, 0)),
                    }
                )
            off_taxonomy = sorted(set(assigned.dropna()) - set(allowed))
            for value in off_taxonomy:
                rows.append(
                    {
                        "label_kind": label_kind,
                        "label_value": value,
                        "n_conversations": int(counts.get(value, 0)),
                        "off_taxonomy": True,
                    }
                )
        out = pd.DataFrame(rows)
        if "off_taxonomy" in out.columns:
            out["off_taxonomy"] = out["off_taxonomy"].fillna(False)
        else:
            out["off_taxonomy"] = False
        return out.sort_values(
            ["label_kind", "n_conversations"], ascending=[True, False], ignore_index=True
        )


@dataclass
class LabelSet:
    """Labels produced for one conversation."""

    conversation_id: str
    values: dict[str, list[str] | str | None] = field(default_factory=dict)
    label_source: str | None = None
    parse_success: bool = True
    raw_response: str | None = None

    def to_rows(self) -> list[dict[str, object]]:
        rows = []
        for label_kind, value in self.values.items():
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
                    continue
                rows.append(
                    {
                        "conversation_id": self.conversation_id,
                        "label_kind": label_kind,
                        "label_value": str(candidate).strip(),
                        "label_source": self.label_source,
                    }
                )
        return rows


class Labeler(ABC):
    """Base class for conversation labelers."""

    #: Label kinds this labeler produces.
    label_kinds: tuple[str, ...] = ()

    @abstractmethod
    def label_one(self, conversation_id: str, text: str) -> LabelSet:
        """Label a single conversation from its concatenated text."""

    def label(self, documents: pd.DataFrame, *, text_column: str = "document") -> pd.DataFrame:
        """Label every row of ``documents`` and return an AUDRE labels table.

        ``documents`` must contain ``conversation_id`` and ``text_column``. Rows
        whose response could not be parsed are still returned via
        :meth:`failures`, so a partial run is never silently truncated.
        """

        for column in ("conversation_id", text_column):
            if column not in documents.columns:
                raise KeyError(f"documents is missing {column!r}")

        rows: list[dict[str, object]] = []
        failures: list[LabelSet] = []

        for record in documents.itertuples(index=False):
            result = self.label_one(
                str(getattr(record, "conversation_id")),
                str(getattr(record, text_column) or ""),
            )
            rows.extend(result.to_rows())
            if not result.parse_success:
                failures.append(result)

        out = pd.DataFrame(
            rows, columns=["conversation_id", "label_kind", "label_value", "label_source"]
        )
        out.attrs["n_failures"] = len(failures)
        out.attrs["failures"] = failures
        return out.drop_duplicates(ignore_index=True)
