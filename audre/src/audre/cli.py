"""Command-line interface for AUDRE.

Exposes the stages that are useful to run standalone:

``audre validate``
    Check input tables against the schema before spending time on analysis.
``audre label``
    Assign labels to a documents table.
``audre analyze``
    Run every analysis layer and write the tidy CSVs the R figure layer reads.
``audre schema``
    Print the minimal input schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import (
    adoption,
    cooccurrence,
    networks,
    resources,
    schema,
    stratify,
    transitions,
)


def _read(path: str | Path, table: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return schema.coerce(frame, schema.SPECS[table])


def _write(frame: pd.DataFrame, directory: Path, name: str) -> None:
    if frame is None or frame.empty:
        print(f"  skipped {name} (no rows)")
        return
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / name, index=False)
    print(f"  wrote {name} ({len(frame):,} rows)")


def cmd_schema(args: argparse.Namespace) -> int:
    frame = schema.describe_schema()
    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    else:
        print(frame.to_string(index=False))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    conversations = _read(args.conversations, "conversations")
    labels = _read(args.labels, "labels")
    resource_frame = _read(args.resources, "resources") if args.resources else None

    reports = schema.validate_all(
        conversations, labels, resource_frame, strict=args.strict
    )
    for report in reports:
        print(report)

    return 0 if all(report.ok for report in reports) else 1


def cmd_label(args: argparse.Namespace) -> int:
    documents = pd.read_csv(args.documents)

    if args.labeler == "rules":
        from .labeling import RuleLabeler

        labeler = RuleLabeler()
    else:
        from .labeling import LLMLabeler

        labeler = LLMLabeler(model=args.model)

    labels = labeler.label(documents, text_column=args.text_column)
    n_failures = labels.attrs.get("n_failures", 0)

    labels.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(labels):,} label rows)")
    if n_failures:
        print(f"WARNING: {n_failures:,} conversation(s) failed to parse", file=sys.stderr)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    conversations = _read(args.conversations, "conversations")
    labels = _read(args.labels, "labels")
    resource_frame = _read(args.resources, "resources") if args.resources else None

    reports = schema.validate_all(conversations, labels, resource_frame)
    for report in reports:
        if not report.ok:
            print(report, file=sys.stderr)
            return 1

    out = Path(args.output_dir)
    kind = args.label_kind
    print(f"Analyzing label_kind={kind!r} -> {out}")

    _write(
        adoption.label_frequency(conversations, labels, label_kind=kind),
        out,
        "label_frequency.csv",
    )
    _write(
        adoption.interaction_depth(conversations, labels, label_kind=kind),
        out,
        "interaction_depth.csv",
    )
    _write(adoption.adoption_over_time(conversations), out, "adoption_over_time.csv")

    if "model_family" in conversations.columns:
        _write(adoption.model_share(conversations), out, "model_share.csv")
        _write(
            adoption.multi_model_adoption(conversations), out, "multi_model_adoption.csv"
        )

    if args.stratify_by in conversations.columns:
        _write(
            stratify.stratum_sizes(conversations, args.stratify_by),
            out,
            "stratum_sizes.csv",
        )
        _write(
            stratify.label_share_matrix(
                conversations, labels, by=args.stratify_by, label_kind=kind
            ),
            out,
            "label_share_by_stratum.csv",
        )

    taxonomy_kinds = set(labels["label_kind"].dropna().unique()) & {
        "work_activity",
        "request_topic",
        "occupation_group",
    }
    if taxonomy_kinds and args.stratify_by in conversations.columns:
        taxonomy_kind = sorted(taxonomy_kinds)[0]
        _write(
            adoption.leading_label_by_unit(
                conversations,
                labels,
                label_kind=taxonomy_kind,
                unit_column=args.stratify_by,
            ),
            out,
            "leading_label_by_unit.csv",
        )

    transition_edges = transitions.build_transitions(
        conversations, labels, label_kind=kind, window_hours=args.window_hours
    )
    _write(transition_edges, out, "transition_edges.csv")
    if not transition_edges.empty:
        _write(transitions.asymmetry(transition_edges), out, "transition_asymmetry.csv")
        _write(
            networks.node_metrics(networks.build_graph(transition_edges)),
            out,
            "transition_node_metrics.csv",
        )

    cooccurrence_edges = cooccurrence.build_cooccurrence(
        conversations, labels, label_kind=kind
    )
    _write(cooccurrence_edges, out, "cooccurrence_edges.csv")
    if not cooccurrence_edges.empty:
        _write(
            networks.node_metrics(networks.build_graph(cooccurrence_edges)),
            out,
            "cooccurrence_node_metrics.csv",
        )

    if resource_frame is not None:
        resolved = resources.resolve_cost(resource_frame)
        _write(resources.cost_coverage(resolved), out, "cost_coverage.csv")
        _write(
            resources.resource_intensity_by_label(
                resolved, labels, label_kind=kind, conversations=conversations
            ),
            out,
            "resource_intensity.csv",
        )

    print("Analysis complete. Render figures with: Rscript R/01_adoption_engagement.R")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audre",
        description=(
            "Turn enterprise GenAI chat telemetry into context-specific usage "
            "representations and candidate workflow hypotheses."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="print the minimal input schema")
    schema_parser.add_argument("--output", help="write the schema to CSV instead of stdout")
    schema_parser.set_defaults(func=cmd_schema)

    validate_parser = subparsers.add_parser("validate", help="validate input tables")
    validate_parser.add_argument("--conversations", required=True)
    validate_parser.add_argument("--labels", required=True)
    validate_parser.add_argument("--resources")
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="treat unrecognized columns as errors",
    )
    validate_parser.set_defaults(func=cmd_validate)

    label_parser = subparsers.add_parser("label", help="assign labels to a documents table")
    label_parser.add_argument("--documents", required=True)
    label_parser.add_argument("--output", required=True)
    label_parser.add_argument("--text-column", default="document")
    label_parser.add_argument(
        "--labeler",
        choices=("rules", "llm"),
        default="rules",
        help="'rules' needs no credentials and is deterministic; 'llm' needs "
        "AUDRE_LLM_BASE_URL and AUDRE_LLM_API_KEY",
    )
    label_parser.add_argument("--model", default="llama_4_maverick")
    label_parser.set_defaults(func=cmd_label)

    analyze_parser = subparsers.add_parser(
        "analyze", help="run all analysis layers and write tidy CSVs for the R figures"
    )
    analyze_parser.add_argument("--conversations", required=True)
    analyze_parser.add_argument("--labels", required=True)
    analyze_parser.add_argument("--resources")
    analyze_parser.add_argument("--output-dir", default="output/tables")
    analyze_parser.add_argument("--label-kind", default="topic_tag")
    analyze_parser.add_argument("--stratify-by", default="unit")
    analyze_parser.add_argument(
        "--window-hours",
        type=float,
        default=transitions.DEFAULT_WINDOW_HOURS,
        help="transition window; an analytical choice, not a property of the data",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
