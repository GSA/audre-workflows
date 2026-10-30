#!/usr/bin/env python3

"""
04_prepare_quarter_partitions.py

Create publication-ready Q1 and Q2 analysis tables from the approved
Jan-Jun 2026 processed USAi conversation tables.

IMPORTANT:
    This script starts from the approved processed data.
    It does NOT require or access the original raw GSA exports.

The script:

1. Reads the Jan-Jun conversation-level processed tables.
2. Uses conversation_created to assign each conversation to:
       Q1 = 2026-01-01 through 2026-03-31
       Q2 = 2026-04-01 through 2026-06-30
3. Uses conversation_id to create consistent subsets across all tables.
4. Replaces official GSA office labels with publication-safe,
   functionally descriptive office names.
5. Writes separate Q1 and Q2 processed tables.
6. Produces a QC summary of the temporal split.

The original Jan-Jun processed files are NEVER modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ==========================================================
# 0. Configuration
# ==========================================================

# Quarter boundaries are defined as half-open intervals:
#
# Q1:
#   2026-01-01 00:00:00 UTC
#       <= date <
#   2026-04-01 00:00:00 UTC
#
# Q2:
#   2026-04-01 00:00:00 UTC
#       <= date <
#   2026-07-01 00:00:00 UTC
#
# Using half-open intervals prevents overlap at quarter boundaries.

QUARTERS = {
    "q1": {
        "start": pd.Timestamp("2026-01-01", tz="UTC"),
        "end": pd.Timestamp("2026-04-01", tz="UTC"),
    },
    "q2": {
        "start": pd.Timestamp("2026-04-01", tz="UTC"),
        "end": pd.Timestamp("2026-07-01", tz="UTC"),
    },
}


# ----------------------------------------------------------
# Publication-safe office names
# ----------------------------------------------------------
#
# These labels preserve broad organizational function while
# removing the official GSA office/service names.
#
# IMPORTANT:
# If an office appears in the processed data but is absent here,
# the script will STOP rather than accidentally write the original
# office name into a publication-ready dataset.

OFFICE_MAP = {
    "FAS": "Acquisition Office",
    "PBS": "Real Estate Office",
    "GSA IT": "Technology Office",
    "OCFO": "Finance Office",
    "OGP": "Policy Office",
    "OSC": "Communications Office",
    "OHRM": "Human Resources Office",
    "OGC": "Legal Office",
    "OMA": "Mission Support Office",
    "OCR": "Civil Rights Office",
    "OSDBU": "Small Business Office",
    "OAS": "Administrative Services Office",
    "OCIA": "External Affairs Office",
    "IOA": "Independent Office",
    "OIG": "Oversight Office",
}


# ==========================================================
# 1. Arguments
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split approved Jan-Jun 2026 conversation tables "
            "into Q1 and Q2 publication-ready datasets."
        )
    )

    parser.add_argument(
        "--processed-dir",
        default="processed_data",
        help=(
            "Directory containing the approved Jan-Jun "
            "processed conversation tables."
        ),
    )

    parser.add_argument(
        "--input-prefix",
        default="employed_jan_jun2026",
        help="Prefix used by the approved Jan-Jun processed files.",
    )

    parser.add_argument(
        "--output-dir",
        default="processed_data/quarter_partitions",
        help="Directory where Q1/Q2 derived datasets will be written.",
    )

    return parser.parse_args()


# ==========================================================
# 2. General helpers
# ==========================================================

def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"Reading: {path}")

    return pd.read_csv(
        path,
        low_memory=False,
    )


def write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Write a CSV and report its dimensions."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Wrote: {path} "
        f"({len(df):,} rows, {len(df.columns):,} columns)"
    )


# ==========================================================
# 3. Validate source tables
# ==========================================================

def validate_model_table(
    df: pd.DataFrame,
) -> None:
    """
    Validate the model-analysis table.

    This table defines the canonical conversation population
    and quarter membership.
    """

    required = {
        "conversation_id",
        "conversation_created",
        "user_id",
        "office",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Model-analysis table is missing required columns: "
            f"{sorted(missing)}"
        )

    duplicate_mask = df["conversation_id"].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        n_duplicates = (
            df.loc[
                duplicate_mask,
                "conversation_id",
            ]
            .nunique()
        )

        raise ValueError(
            "Expected one row per conversation in the "
            "model-analysis table, but found "
            f"{n_duplicates:,} duplicated conversation IDs."
        )


# ==========================================================
# 4. Parse timestamps
# ==========================================================

def parse_conversation_dates(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Parse conversation_created as timezone-aware UTC timestamps.

    Example source value:

        2026-01-02 02:28:08+00:00
    """

    dates = pd.to_datetime(
        df["conversation_created"],
        errors="coerce",
        utc=True,
    )

    n_missing = dates.isna().sum()

    if n_missing > 0:
        print(
            f"WARNING: {n_missing:,} conversations have an "
            "unparseable or missing conversation_created value."
        )

    return dates


# ==========================================================
# 5. Office anonymization
# ==========================================================

def validate_office_mapping(
    df: pd.DataFrame,
) -> None:
    """
    Ensure every non-missing source office has an anonymized label.
    """

    source_offices = set(
        df["office"]
        .dropna()
        .astype(str)
        .unique()
    )

    mapped_offices = set(
        OFFICE_MAP.keys()
    )

    unmapped = sorted(
        source_offices - mapped_offices
    )

    if unmapped:
        raise ValueError(
            "\nThe following offices are present in the data "
            "but missing from OFFICE_MAP:\n\n"
            + "\n".join(
                f"  - {office}"
                for office in unmapped
            )
            + "\n\nAdd them to OFFICE_MAP before continuing."
        )


def anonymize_offices(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace official office labels with publication-safe labels.

    Original office names are NOT retained in the output.
    """

    if "office" not in df.columns:
        return df.copy()

    result = df.copy()

    original = result["office"].astype("string")

    result["office"] = original.map(
        OFFICE_MAP
    )

    # Preserve true missing values.
    result.loc[
        original.isna(),
        "office",
    ] = pd.NA

    # Safety check.
    unmapped_mask = (
        original.notna()
        & result["office"].isna()
    )

    if unmapped_mask.any():
        unmapped = sorted(
            original.loc[
                unmapped_mask
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Office anonymization failed for: "
            + ", ".join(unmapped)
        )

    return result


# ==========================================================
# 6. Quarter membership
# ==========================================================

def get_quarter_ids(
    model_df: pd.DataFrame,
    conversation_dates: pd.Series,
    quarter: str,
) -> set[str]:
    """
    Return conversation IDs belonging to the requested quarter.
    """

    start = QUARTERS[quarter]["start"]
    end = QUARTERS[quarter]["end"]

    mask = (
        (conversation_dates >= start)
        & (conversation_dates < end)
    )

    return set(
        model_df.loc[
            mask,
            "conversation_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )


def subset_by_conversation_id(
    df: pd.DataFrame,
    conversation_ids: set[str],
) -> pd.DataFrame:
    """
    Subset any processed table using canonical conversation IDs.
    """

    if "conversation_id" not in df.columns:
        raise ValueError(
            "Expected a conversation_id column."
        )

    ids = (
        df["conversation_id"]
        .astype(str)
    )

    return (
        df.loc[
            ids.isin(conversation_ids)
        ]
        .copy()
    )


# ==========================================================
# 7. Quarter-level QC
# ==========================================================

def make_quarter_qc_row(
    quarter: str,
    model_df: pd.DataFrame,
    tags_df: pd.DataFrame,
) -> dict:
    """Create one row of quarter-level summary statistics."""

    return {
        "quarter": quarter.upper(),

        "start_date": (
            QUARTERS[quarter]["start"]
            .date()
        ),

        "end_date_exclusive": (
            QUARTERS[quarter]["end"]
            .date()
        ),

        "n_conversations": (
            model_df["conversation_id"]
            .nunique()
        ),

        "n_users": (
            model_df["user_id"]
            .nunique()
            if "user_id" in model_df.columns
            else pd.NA
        ),

        "n_offices": (
            model_df["office"]
            .nunique()
            if "office" in model_df.columns
            else pd.NA
        ),

        "n_tag_rows": len(
            tags_df
        ),

        "n_unique_tags": (
            tags_df["tag"]
            .nunique()
            if "tag" in tags_df.columns
            else pd.NA
        ),

        "estimated_total_tokens": (
            model_df[
                "estimated_total_tokens"
            ]
            .sum()
            if "estimated_total_tokens"
            in model_df.columns
            else pd.NA
        ),

        "estimated_total_cost_usd": (
            model_df[
                "estimated_total_cost_usd"
            ]
            .sum()
            if "estimated_total_cost_usd"
            in model_df.columns
            else pd.NA
        ),
    }


# ==========================================================
# 8. Main
# ==========================================================

def main() -> None:

    args = parse_args()

    processed_dir = Path(
        args.processed_dir
    )

    output_root = Path(
        args.output_dir
    )

    prefix = args.input_prefix

    # ------------------------------------------------------
    # Locate approved Jan-Jun inputs
    # ------------------------------------------------------

    model_path = (
        processed_dir
        / f"{prefix}_conversation_model_analysis.csv"
    )

    tags_path = (
        processed_dir
        / f"{prefix}_conversation_tags.csv"
    )

    # ------------------------------------------------------
    # Read source tables
    # ------------------------------------------------------

    model_df = read_csv(
        model_path
    )

    tags_df = read_csv(
        tags_path
    )

    # ------------------------------------------------------
    # Validate source population
    # ------------------------------------------------------

    validate_model_table(
        model_df
    )

    validate_office_mapping(
        model_df
    )

    conversation_dates = (
        parse_conversation_dates(
            model_df
        )
    )

    print("\n" + "=" * 60)
    print("SOURCE DATA")
    print("=" * 60)

    print(
        f"Model-analysis rows: "
        f"{len(model_df):,}"
    )

    print(
        "Unique conversations: "
        f"{model_df['conversation_id'].nunique():,}"
    )

    print(
        "Unique users: "
        f"{model_df['user_id'].nunique():,}"
    )

    print(
        "Date range: "
        f"{conversation_dates.min()} "
        f"through "
        f"{conversation_dates.max()}"
    )

    # ------------------------------------------------------
    # Show anonymization mapping
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("OFFICE ANONYMIZATION")
    print("=" * 60)

    office_counts = (
        model_df["office"]
        .value_counts(
            dropna=False
        )
    )

    for office, count in office_counts.items():

        if pd.isna(office):
            public_name = "<missing>"

        else:
            public_name = OFFICE_MAP[
                str(office)
            ]

        print(
            f"{str(office):<10} "
            f"-> "
            f"{public_name:<28} "
            f"{count:>8,} conversations"
        )

    # ------------------------------------------------------
    # Create quarter outputs
    # ------------------------------------------------------

    quarter_id_sets = {}
    qc_rows = []

    for quarter in ["q1", "q2"]:

        print("\n" + "=" * 60)
        print(
            f"PREPARING {quarter.upper()}"
        )
        print("=" * 60)

        conversation_ids = (
            get_quarter_ids(
                model_df,
                conversation_dates,
                quarter,
            )
        )

        quarter_id_sets[
            quarter
        ] = conversation_ids

        # ----------------------------------------------
        # Subset all tables using the SAME conversation IDs
        # ----------------------------------------------

        q_model = (
            subset_by_conversation_id(
                model_df,
                conversation_ids,
            )
        )

        q_tags = (
            subset_by_conversation_id(
                tags_df,
                conversation_ids,
            )
        )

        # ----------------------------------------------
        # Apply publication-safe office names
        # ----------------------------------------------

        q_model = anonymize_offices(
            q_model
        )

        q_tags = anonymize_offices(
            q_tags
        )

        # ----------------------------------------------
        # Output locations
        # ----------------------------------------------

        quarter_prefix = (
            f"employed_{quarter}_2026"
        )

        quarter_dir = (
            output_root / quarter
        )

        # ----------------------------------------------
        # Write quarter-specific processed tables
        # ----------------------------------------------

        write_csv(
            q_model,
            quarter_dir
            / (
                f"{quarter_prefix}"
                "_conversation_model_analysis.csv"
            ),
        )

        write_csv(
            q_tags,
            quarter_dir
            / (
                f"{quarter_prefix}"
                "_conversation_tags.csv"
            ),
        )

        # ----------------------------------------------
        # Quarter QC
        # ----------------------------------------------

        qc_rows.append(
            make_quarter_qc_row(
                quarter,
                q_model,
                q_tags,
            )
        )

    # ======================================================
    # 9. Cross-quarter QC
    # ======================================================

    overlap = (
        quarter_id_sets["q1"]
        & quarter_id_sets["q2"]
    )

    if overlap:
        raise ValueError(
            f"{len(overlap):,} conversation IDs "
            "were assigned to BOTH Q1 and Q2."
        )

    union = (
        quarter_id_sets["q1"]
        | quarter_id_sets["q2"]
    )

    source_ids = set(
        model_df[
            "conversation_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    unassigned = (
        source_ids - union
    )

    # ------------------------------------------------------
    # Distinguish invalid timestamps from dates outside range
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-QUARTER QC")
    print("=" * 60)

    print(
        f"Q1 conversations: "
        f"{len(quarter_id_sets['q1']):,}"
    )

    print(
        f"Q2 conversations: "
        f"{len(quarter_id_sets['q2']):,}"
    )

    print(
        f"Q1/Q2 overlap: "
        f"{len(overlap):,}"
    )

    print(
        f"Unassigned source conversations: "
        f"{len(unassigned):,}"
    )

    # ------------------------------------------------------
    # Write QC table
    # ------------------------------------------------------

    qc_df = pd.DataFrame(
        qc_rows
    )

    qc_path = (
        output_root
        / "quarter_partition_summary.csv"
    )

    write_csv(
        qc_df,
        qc_path,
    )

    print("\nQuarter partition summary:")
    print(
        qc_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Final warning if anything was not assigned
    # ------------------------------------------------------

    if unassigned:
        print(
            "\nWARNING:"
            "\nSome source conversations were not assigned "
            "to Q1 or Q2."
            "\nThis can occur if timestamps are missing or "
            "fall outside Jan-Jun 2026."
        )

    else:
        print(
            "\nAll source conversations were assigned "
            "to exactly one quarter."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()