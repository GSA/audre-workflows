"""Privacy-preserving preparation and reporting rules.

AUDRE treats privacy as a pipeline stage rather than a manual review step, so
that the same protections apply every time an analysis is re-run.

Three mechanisms are provided:

1. :func:`deidentify` — replace direct identifiers with salted pseudonyms and
   drop conversation content before analysis begins.
2. :func:`suppress_small_cells` — enforce minimum conversation *and* minimum
   distinct-user counts on any aggregate before it is reported.
3. :func:`assert_reportable` — a hard gate for figures and tables, so a
   suppression rule cannot be forgotten on the way to a paper.

The defaults are deliberately conservative. A cell must be supported by at
least ``MIN_CONVERSATIONS`` conversations contributed by at least
``MIN_USERS`` distinct users; requiring distinct users matters because a
single prolific user can otherwise make an aggregate effectively personal.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

import pandas as pd

from .schema import FORBIDDEN_COLUMNS

MIN_CONVERSATIONS = 5
MIN_USERS = 3

SUPPRESSED = "suppressed_small_cell"


def new_salt(n_bytes: int = 32) -> str:
    """Generate a fresh pseudonymization salt.

    Generate the salt once per analysis, keep it out of version control, and
    discard it when re-linking is no longer required. Discarding the salt makes
    the pseudonyms non-reversible even by the analyst.
    """

    return secrets.token_hex(n_bytes)


def pseudonymize(values: pd.Series, salt: str, *, prefix: str = "u", length: int = 12) -> pd.Series:
    """Map identifiers to stable salted pseudonyms.

    Uses HMAC-SHA256 rather than a bare hash so that the salt acts as a key: an
    adversary holding a list of candidate identifiers cannot confirm membership
    without also holding the salt.
    """

    if not salt:
        raise ValueError("A non-empty salt is required; see audre.privacy.new_salt().")

    key = salt.encode("utf-8")

    def digest(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        mac = hmac.new(key, str(value).encode("utf-8"), hashlib.sha256)
        return f"{prefix}_{mac.hexdigest()[:length]}"

    return values.map(digest).astype("string")


def deidentify(
    conversations: pd.DataFrame,
    *,
    salt: str,
    id_columns: tuple[str, ...] = ("user_id",),
    unit_map: dict[str, str] | None = None,
    drop_content: bool = True,
) -> pd.DataFrame:
    """Pseudonymize identifiers, optionally relabel units, and drop content.

    Parameters
    ----------
    salt:
        Pseudonymization key from :func:`new_salt`.
    id_columns:
        Columns to replace with pseudonyms.
    unit_map:
        Optional mapping from real organizational unit names to neutral labels
        (for example ``{"Facilities Office": "Unit B"}``). Units absent from
        the map are passed through, so supply a complete map when preparing a
        dataset for external release.
    drop_content:
        Remove any column carrying conversation text or direct identifiers.
    """

    out = conversations.copy()

    for column in id_columns:
        if column in out.columns:
            out[column] = pseudonymize(out[column], salt=salt)

    if unit_map is not None and "unit" in out.columns:
        out["unit"] = out["unit"].map(lambda value: unit_map.get(value, value))

    if drop_content:
        present = [column for column in FORBIDDEN_COLUMNS if column in out.columns]
        out = out.drop(columns=present)

    return out


def suppress_small_cells(
    frame: pd.DataFrame,
    *,
    count_column: str = "n_conversations",
    user_column: str | None = "n_users",
    min_conversations: int = MIN_CONVERSATIONS,
    min_users: int = MIN_USERS,
    mode: str = "drop",
    label_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply minimum-cell rules to an aggregate table.

    Parameters
    ----------
    mode:
        ``"drop"`` removes failing rows. ``"mask"`` keeps them but blanks the
        counts and rewrites ``label_columns`` to :data:`SUPPRESSED`, which
        preserves the fact that a category existed without revealing its size.
    """

    if mode not in {"drop", "mask"}:
        raise ValueError("mode must be 'drop' or 'mask'")
    if count_column not in frame.columns:
        raise KeyError(f"count column {count_column!r} not found in frame")

    out = frame.copy()
    keep = out[count_column].fillna(0) >= min_conversations

    if user_column and user_column in out.columns:
        keep &= out[user_column].fillna(0) >= min_users
    elif user_column:
        raise KeyError(
            f"user column {user_column!r} not found. Pass user_column=None only when "
            "the aggregate genuinely cannot be attributed to users."
        )

    out.attrs["n_suppressed"] = int((~keep).sum())
    out.attrs["suppression_rule"] = (
        f"{count_column} >= {min_conversations} and "
        f"{user_column} >= {min_users}" if user_column else f"{count_column} >= {min_conversations}"
    )

    if mode == "drop":
        return out[keep].reset_index(drop=True)

    numeric = out.select_dtypes(include="number").columns
    out.loc[~keep, numeric] = pd.NA
    for column in label_columns:
        if column in out.columns:
            out[column] = out[column].astype("string")
            out.loc[~keep, column] = SUPPRESSED
    return out


def assert_reportable(
    frame: pd.DataFrame,
    *,
    count_column: str = "n_conversations",
    user_column: str | None = "n_users",
    min_conversations: int = MIN_CONVERSATIONS,
    min_users: int = MIN_USERS,
) -> pd.DataFrame:
    """Raise unless every row of ``frame`` satisfies the minimum-cell rules.

    Call this immediately before writing a figure or table. It fails loudly so
    that an un-suppressed aggregate cannot reach a publication by accident.
    """

    failures = frame[count_column].fillna(0) < min_conversations
    if user_column and user_column in frame.columns:
        failures |= frame[user_column].fillna(0) < min_users

    n_failures = int(failures.sum())
    if n_failures:
        raise ValueError(
            f"{n_failures:,} row(s) fall below the reporting threshold "
            f"({count_column} >= {min_conversations}"
            + (f", {user_column} >= {min_users}" if user_column else "")
            + "). Apply audre.privacy.suppress_small_cells() before reporting."
        )
    return frame


def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness per column.

    Reported alongside results because suppression and missingness interact:
    a column that is 60% missing can make a surviving cell unrepresentative
    even when it clears the count thresholds.
    """

    n_rows = len(frame)
    rows = [
        {
            "column": column,
            "n_missing": int(frame[column].isna().sum()),
            "share_missing": float(frame[column].isna().mean()) if n_rows else 0.0,
            "n_distinct": int(frame[column].nunique(dropna=True)),
        }
        for column in frame.columns
    ]
    return pd.DataFrame(rows).sort_values("share_missing", ascending=False, ignore_index=True)
