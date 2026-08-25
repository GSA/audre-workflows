"""AUDRE — Analyzing Usage, Depth, Resource intensity, and Engagement.

A reproducible, privacy-aware pipeline for turning enterprise GenAI chat
telemetry into context-specific usage representations and candidate workflow
hypotheses.

The pipeline is organized as stages, and the ordering is deliberate: privacy
preparation happens before analysis, and every analysis stage emits aggregates
that have already passed a minimum-cell rule.

    preprocess  ->  privacy  ->  labeling  ->  analysis  ->  figures (R)

The analysis stages correspond to the acronym:

``adoption``
    Usage volume and Engagement depth. Volume and depth rank activities
    differently, so both are reported.
``resources``
    Resource intensity in tokens and estimated cost, with observed versus
    estimated provenance kept separate.
``transitions``
    Directed relationships between temporally adjacent conversations.
``cooccurrence`` / ``networks``
    Undirected same-conversation relationships, and graph metrics for both.
``stratify``
    Any of the above computed within organizational or model context.

Interpretation contract
-----------------------
AUDRE distinguishes what is **recorded** (users, conversations, messages,
timestamps, models) from what is **assigned** (labels), **derived**
(transitions, networks), and **inferred** (candidate workflows). Outcomes —
productivity, quality, time saved, cost avoided, return on investment — are
none of these. They require independently measured outcomes and a baseline or
counterfactual, and nothing in this package estimates them.
"""

from . import (
    adoption,
    cooccurrence,
    labeling,
    networks,
    preprocess,
    privacy,
    resources,
    schema,
    stratify,
    transitions,
)

__version__ = "0.1.0"

__all__ = [
    "adoption",
    "cooccurrence",
    "labeling",
    "networks",
    "preprocess",
    "privacy",
    "resources",
    "schema",
    "stratify",
    "transitions",
    "__version__",
]
