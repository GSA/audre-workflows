"""End-to-end AUDRE run, from labeling through the tables the R figures read.

Demonstrates the intended stage ordering. Run from the ``audre/`` directory:

    python examples/run_pipeline.py

By default it uses the committed test fixture, which exists only to exercise
pipeline wiring — it is far too small and too regular to support any finding.
Point ``--conversations``/``--labels``/``--resources`` at real tables to use it
for an actual analysis.

Then render the figures:

    AUDRE_TABLE_DIR=examples/output/tables \\
    AUDRE_PLOT_DIR=examples/output/plots \\
    Rscript R/01_adoption_engagement.R
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from audre import (
    adoption,
    cooccurrence,
    labeling,
    networks,
    privacy,
    resources,
    schema,
    stratify,
    transitions,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "tests" / "fixtures"

LABEL_KIND = "topic_tag"
TAXONOMY_KIND = "work_activity"
STRATIFY_BY = "unit"


def write(frame: pd.DataFrame, directory: Path, name: str) -> None:
    if frame is None or frame.empty:
        print(f"    (skipped {name}: no rows survived)")
        return
    frame.to_csv(directory / name, index=False)
    print(f"    {name:38s} {len(frame):>6,} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", default=FIXTURES / "conversations.csv")
    parser.add_argument("--labels", default=FIXTURES / "labels.csv")
    parser.add_argument("--resources", default=FIXTURES / "resources.csv")
    parser.add_argument("--documents", default=FIXTURES / "documents.csv")
    parser.add_argument("--output-dir", default=HERE / "output")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=transitions.DEFAULT_WINDOW_HOURS,
        help="transition window; an analytical choice, not a property of the data",
    )
    args = parser.parse_args()

    tables = Path(args.output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # -- 1. Load and validate ------------------------------------------------
    # Validation runs before anything expensive, so a schema problem surfaces
    # immediately rather than as a confusing result several stages later.
    print("\n[1] Load and validate")
    conversations = schema.coerce(pd.read_csv(args.conversations), schema.CONVERSATIONS)
    labels = schema.coerce(pd.read_csv(args.labels), schema.LABELS)
    resource_records = schema.coerce(pd.read_csv(args.resources), schema.RESOURCES)

    for report in schema.validate_all(conversations, labels, resource_records):
        print(f"    {report}".replace("\n", "\n    "))
        report.raise_if_invalid()

    # -- 2. Privacy preparation ---------------------------------------------
    # Pseudonymization happens before analysis, not before publication, so that
    # no intermediate artifact ever holds a direct identifier. Generate the salt
    # per analysis and keep it out of version control.
    print("\n[2] Privacy preparation")
    salt = privacy.new_salt()
    conversations = privacy.deidentify(conversations, salt=salt)
    print(f"    pseudonymized {conversations['user_id'].nunique():,} users")
    print(
        f"    reporting rule: n_conversations >= {privacy.MIN_CONVERSATIONS}, "
        f"n_users >= {privacy.MIN_USERS}"
    )

    # -- 3. Labeling ---------------------------------------------------------
    # Shown with the deterministic rule labeler so this script runs with no
    # credentials. Swap in labeling.LLMLabeler() for the LLM path.
    print("\n[3] Labeling (demonstration only; using the fixture's own labels below)")
    documents = pd.read_csv(args.documents)
    demo_labels = labeling.RuleLabeler().label(documents.head(25))
    print(
        f"    rule labeler produced {len(demo_labels):,} label rows "
        f"for {demo_labels['conversation_id'].nunique():,} conversations"
    )
    coverage = labeling.Taxonomy.load().coverage(labels)
    write(coverage, tables, "taxonomy_coverage.csv")

    # -- 4. Adoption and engagement -----------------------------------------
    print("\n[4] Adoption and engagement")
    write(
        adoption.label_frequency(conversations, labels, label_kind=LABEL_KIND),
        tables,
        "label_frequency.csv",
    )
    write(
        adoption.interaction_depth(conversations, labels, label_kind=LABEL_KIND),
        tables,
        "interaction_depth.csv",
    )
    write(adoption.adoption_over_time(conversations), tables, "adoption_over_time.csv")
    write(adoption.model_share(conversations), tables, "model_share.csv")
    write(
        adoption.multi_model_adoption(conversations), tables, "multi_model_adoption.csv"
    )

    # -- 5. Context stratification ------------------------------------------
    print("\n[5] Context stratification")
    write(stratify.stratum_sizes(conversations, STRATIFY_BY), tables, "stratum_sizes.csv")
    write(
        stratify.label_share_matrix(
            conversations, labels, by=STRATIFY_BY, label_kind=LABEL_KIND
        ),
        tables,
        "label_share_by_stratum.csv",
    )
    write(
        adoption.leading_label_by_unit(
            conversations, labels, label_kind=TAXONOMY_KIND, unit_column=STRATIFY_BY
        ),
        tables,
        "leading_label_by_unit.csv",
    )

    # -- 6. Directed transitions --------------------------------------------
    print("\n[6] Directed transitions")
    transition_edges = transitions.build_transitions(
        conversations, labels, label_kind=LABEL_KIND, window_hours=args.window_hours
    )
    write(transition_edges, tables, "transition_edges.csv")

    if not transition_edges.empty:
        write(transitions.asymmetry(transition_edges), tables, "transition_asymmetry.csv")
        write(
            networks.node_metrics(networks.build_graph(transition_edges)),
            tables,
            "transition_node_metrics.csv",
        )

    # The window is an analyst-chosen parameter, so its influence is reported
    # rather than assumed away.
    write(
        transitions.window_sweep(
            conversations, labels, label_kind=LABEL_KIND, windows_hours=(6.0, 12.0, 24.0, 48.0)
        ),
        tables,
        "transition_window_sweep.csv",
    )

    # Per-unit networks, replacing what used to be one hardcoded script per unit.
    write(
        stratify.by_stratum(
            transitions.build_transitions,
            conversations,
            labels,
            by=STRATIFY_BY,
            label_kind=LABEL_KIND,
            window_hours=args.window_hours,
        ),
        tables,
        "transition_edges_by_stratum.csv",
    )

    # -- 7. Co-occurrence ----------------------------------------------------
    print("\n[7] Co-occurrence")
    cooccurrence_edges = cooccurrence.build_cooccurrence(
        conversations, labels, label_kind=LABEL_KIND
    )
    write(cooccurrence_edges, tables, "cooccurrence_edges.csv")
    if not cooccurrence_edges.empty:
        write(
            networks.node_metrics(networks.build_graph(cooccurrence_edges)),
            tables,
            "cooccurrence_node_metrics.csv",
        )

    # -- 8. Resource intensity ----------------------------------------------
    print("\n[8] Resource intensity")
    resolved = resources.resolve_cost(resource_records)
    write(resources.cost_coverage(resolved), tables, "cost_coverage.csv")
    write(
        resources.resource_intensity_by_label(
            resolved, labels, label_kind=LABEL_KIND, conversations=conversations
        ),
        tables,
        "resource_intensity.csv",
    )
    write(privacy.missingness_report(resolved), tables, "resource_missingness.csv")

    print(f"\nTables written to {tables}")
    print(
        "Render figures with:\n"
        f"    AUDRE_TABLE_DIR={tables} AUDRE_PLOT_DIR={Path(args.output_dir) / 'plots'} \\\n"
        "        Rscript R/01_adoption_engagement.R"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
