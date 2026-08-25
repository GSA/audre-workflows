"""Minimal input schema for AUDRE.

AUDRE deliberately does not require raw prompts, responses, conversation
titles, or any user-identifying attribute. Everything downstream is computed
from three narrow tables, so an organization can share or reproduce an
analysis without exporting conversation content.

Tables
------
``conversations``
    One row per chat thread. The spine of every analysis.
``labels``
    Long format, one row per (conversation, label_kind, label_value). Long
    format is required because a conversation may carry several unordered
    topic tags, while taxonomy assignments are single-valued.
``resources``
    Optional. One row per conversation with token and cost measures.

Notes
-----
``unit`` is the organizational-context column (a team, office, or business
line). It is intentionally opaque: AUDRE never interprets it, so callers can
substitute pseudonymous unit names before the data ever reaches the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Label kinds AUDRE understands. ``topic_tag`` is multi-valued per
#: conversation; every other kind is expected to be single-valued.
LABEL_KINDS = (
    "topic_tag",
    "work_activity",
    "request_topic",
    "occupation_group",
    "use_case_title",
)

MULTI_VALUED_LABEL_KINDS = ("topic_tag",)

#: Columns that must never appear in an AUDRE input table, because they carry
#: conversation content or directly identifying attributes.
FORBIDDEN_COLUMNS = (
    "content",
    "message_text",
    "prompt",
    "response",
    "chat",
    "chat_parsed",
    "email",
    "employee_email",
    "full_name",
    "title",
    "conversation_title",
)


@dataclass(frozen=True)
class TableSpec:
    """Required and optional columns for one AUDRE input table."""

    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    unique_key: tuple[str, ...] = ()

    @property
    def known_columns(self) -> tuple[str, ...]:
        return self.required + self.optional


CONVERSATIONS = TableSpec(
    name="conversations",
    required=("conversation_id", "user_id", "created_at"),
    optional=("unit", "n_messages", "n_user_messages", "model", "model_family"),
    datetime_columns=("created_at",),
    numeric_columns=("n_messages", "n_user_messages"),
    unique_key=("conversation_id",),
)

LABELS = TableSpec(
    name="labels",
    required=("conversation_id", "label_kind", "label_value"),
    optional=("label_source", "label_confidence"),
    numeric_columns=("label_confidence",),
    unique_key=("conversation_id", "label_kind", "label_value"),
)

RESOURCES = TableSpec(
    name="resources",
    required=("conversation_id",),
    optional=(
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "cost_source",
    ),
    numeric_columns=(
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
    ),
    unique_key=("conversation_id",),
)

SPECS = {spec.name: spec for spec in (CONVERSATIONS, LABELS, RESOURCES)}


@dataclass
class ValidationReport:
    """Outcome of validating one table against its :class:`TableSpec`."""

    table: str
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missingness: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> "ValidationReport":
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ValueError(f"Invalid `{self.table}` table:\n  - {joined}")
        return self

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"table": self.table, "severity": "error", "message": message}
            for message in self.errors
        ]
        rows += [
            {"table": self.table, "severity": "warning", "message": message}
            for message in self.warnings
        ]
        return pd.DataFrame(rows, columns=["table", "severity", "message"])

    def __str__(self) -> str:
        status = "OK" if self.ok else "FAILED"
        lines = [f"[{status}] {self.table}: {self.n_rows:,} rows"]
        lines += [f"  error:   {message}" for message in self.errors]
        lines += [f"  warning: {message}" for message in self.warnings]
        return "\n".join(lines)


def coerce(frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Return a copy of ``frame`` with spec dtypes applied.

    Timestamps become timezone-aware UTC, numeric columns become floats, and
    identifier/label columns are stripped of surrounding whitespace. Coercion
    is separated from validation so that callers can inspect what failed to
    parse rather than having values silently dropped.
    """

    out = frame.copy()

    for column in ("conversation_id", "user_id", "unit", "label_kind", "label_value"):
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()

    for column in spec.datetime_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)

    for column in spec.numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def validate(frame: pd.DataFrame, table: str, *, strict: bool = False) -> ValidationReport:
    """Validate ``frame`` against the named spec.

    Parameters
    ----------
    frame:
        Table to check. Should already have been passed through :func:`coerce`.
    table:
        One of ``"conversations"``, ``"labels"``, ``"resources"``.
    strict:
        When true, unrecognized columns are reported as errors rather than
        warnings. Useful in tests and in CI for shared datasets.
    """

    if table not in SPECS:
        raise KeyError(f"Unknown table {table!r}; expected one of {sorted(SPECS)}")

    spec = SPECS[table]
    report = ValidationReport(table=table, n_rows=len(frame))

    missing = [column for column in spec.required if column not in frame.columns]
    if missing:
        report.errors.append(f"missing required column(s): {missing}")
        return report

    leaked = sorted(set(frame.columns) & set(FORBIDDEN_COLUMNS))
    if leaked:
        report.errors.append(
            f"content or identifying column(s) present: {leaked}. "
            "Remove these before analysis; see audre.privacy.deidentify."
        )

    extra = [column for column in frame.columns if column not in spec.known_columns]
    if extra:
        message = f"unrecognized column(s) ignored by AUDRE: {extra}"
        (report.errors if strict else report.warnings).append(message)

    for column in spec.required:
        n_null = int(frame[column].isna().sum())
        if n_null:
            report.errors.append(f"{column} has {n_null:,} missing value(s)")

    if spec.unique_key and all(k in frame.columns for k in spec.unique_key):
        n_duplicated = int(frame.duplicated(subset=list(spec.unique_key)).sum())
        if n_duplicated:
            report.errors.append(
                f"{n_duplicated:,} duplicate row(s) on key {list(spec.unique_key)}"
            )

    if table == "labels" and "label_kind" in frame.columns:
        unknown = sorted(set(frame["label_kind"].dropna()) - set(LABEL_KINDS))
        if unknown:
            report.warnings.append(
                f"non-standard label_kind value(s): {unknown}. "
                "These pass through unchanged but have no built-in semantics."
            )
        single_valued = frame[~frame["label_kind"].isin(MULTI_VALUED_LABEL_KINDS)]
        offending = single_valued.groupby(
            ["conversation_id", "label_kind"], dropna=False
        ).size()
        offending = offending[offending > 1]
        if len(offending):
            report.warnings.append(
                f"{len(offending):,} (conversation, label_kind) pair(s) carry more than "
                "one value for a single-valued kind; downstream shares will not sum to 1"
            )

    for column in spec.known_columns:
        if column in frame.columns and len(frame):
            report.missingness[column] = float(frame[column].isna().mean())

    return report


def validate_all(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    resources: pd.DataFrame | None = None,
    *,
    strict: bool = False,
) -> list[ValidationReport]:
    """Validate every table and check referential integrity across them."""

    reports = [
        validate(conversations, "conversations", strict=strict),
        validate(labels, "labels", strict=strict),
    ]
    if resources is not None:
        reports.append(validate(resources, "resources", strict=strict))

    if "conversation_id" in conversations.columns:
        known = set(conversations["conversation_id"].dropna())
        for frame, report in zip([labels, resources], reports[1:]):
            if frame is None or "conversation_id" not in frame.columns:
                continue
            orphans = set(frame["conversation_id"].dropna()) - known
            if orphans:
                report.warnings.append(
                    f"{len(orphans):,} conversation_id(s) absent from `conversations`; "
                    "these rows are dropped by analyses that join on the spine"
                )

    return reports


def describe_schema() -> pd.DataFrame:
    """Return the schema as a tidy frame, for documentation and papers."""

    rows = []
    for spec in SPECS.values():
        for column in spec.required:
            rows.append({"table": spec.name, "column": column, "requirement": "required"})
        for column in spec.optional:
            rows.append({"table": spec.name, "column": column, "requirement": "optional"})
    return pd.DataFrame(rows)
