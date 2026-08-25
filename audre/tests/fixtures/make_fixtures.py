"""Regenerate the committed test fixtures.

Deterministic and intentionally simple: the fixtures are meant to be readable by
eye so that a failing test can be diagnosed by looking at the input. This is a
wiring fixture, not a simulation study.

Run from the repository's ``audre/`` directory:

    python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

SEED = 20260825

UNITS = ("Unit A", "Unit B", "Unit C")

# Per unit: the topic tags its users draw from, and a pair that tends to occur
# in sequence. The sequence exists so the transition layer has something to
# find; it is not calibrated to any real workflow.
UNIT_PROFILE = {
    "Unit A": {
        "tags": ("software development", "infrastructure", "documentation"),
        "sequence": ("software development", "documentation"),
    },
    "Unit B": {
        "tags": ("procurement", "contract management", "budget"),
        "sequence": ("procurement", "contract management"),
    },
    "Unit C": {
        "tags": ("communication", "policy", "training"),
        "sequence": ("policy", "communication"),
    },
}

# Tags whose conversations run long, so volume and depth rankings differ.
DEEP_TAGS = frozenset({"infrastructure", "contract management"})

MODELS = ("model-alpha-1", "model-beta-2", "model-gamma-1")


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)

    conversation_rows = []
    label_rows = []
    resource_rows = []
    document_rows = []

    start = pd.Timestamp("2026-01-05 13:00:00", tz="UTC")
    conversation_number = 0

    # 12 users per unit clears the stratum-eligibility gate (>= 10 users).
    for unit in UNITS:
        profile = UNIT_PROFILE[unit]
        for user_index in range(12):
            user_id = f"{unit.replace(' ', '').lower()}_u{user_index:02d}"
            # Vary activity so users differ in volume, as they do in practice.
            n_days = 4 + (user_index % 4)

            for day in range(n_days):
                day_start = start + pd.Timedelta(days=day * 2, hours=user_index % 5)

                # A same-day pair, to give the transition layer a signal.
                first_tag, second_tag = profile["sequence"]
                pair = [
                    (day_start, first_tag),
                    (day_start + pd.Timedelta(hours=2 + (user_index % 6)), second_tag),
                ]
                # An unrelated conversation well outside a 24h window, so the
                # window boundary is actually exercised.
                if day % 2 == 0:
                    pair.append(
                        (
                            day_start + pd.Timedelta(hours=40),
                            rng.choice(profile["tags"]),
                        )
                    )

                for created_at, primary_tag in pair:
                    conversation_number += 1
                    conversation_id = f"c{conversation_number:05d}"

                    depth = rng.randint(14, 26) if primary_tag in DEEP_TAGS else rng.randint(2, 6)
                    model = rng.choice(MODELS)

                    conversation_rows.append(
                        {
                            "conversation_id": conversation_id,
                            "user_id": user_id,
                            "created_at": created_at.isoformat(),
                            "unit": unit,
                            "n_messages": depth,
                            "n_user_messages": max(1, depth // 2),
                            "model": model,
                            "model_family": model.split("-")[1],
                        }
                    )

                    tags = [primary_tag]
                    # Some conversations carry a second tag, because multi-label
                    # conversations are a real property the layers must handle.
                    if rng.random() < 0.35:
                        other = rng.choice(
                            [tag for tag in profile["tags"] if tag != primary_tag]
                        )
                        tags.append(other)

                    for tag in tags:
                        label_rows.append(
                            {
                                "conversation_id": conversation_id,
                                "label_kind": "topic_tag",
                                "label_value": tag,
                                "label_source": "fixture",
                            }
                        )

                    label_rows.append(
                        {
                            "conversation_id": conversation_id,
                            "label_kind": "work_activity",
                            "label_value": WORK_ACTIVITY[primary_tag],
                            "label_source": "fixture",
                        }
                    )

                    prompt_tokens = depth * rng.randint(120, 400)
                    completion_tokens = depth * rng.randint(200, 700)
                    # Roughly 70% observed, so provenance handling is exercised.
                    has_observed = rng.random() < 0.7
                    resource_rows.append(
                        {
                            "conversation_id": conversation_id,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                            "cost_usd": (
                                round(
                                    (prompt_tokens * 0.5 + completion_tokens * 2.0) / 1e6,
                                    6,
                                )
                                if has_observed
                                else ""
                            ),
                        }
                    )

                    document_rows.append(
                        {
                            "conversation_id": conversation_id,
                            "document": DOCUMENT_TEXT[primary_tag],
                        }
                    )

    return (
        pd.DataFrame(conversation_rows),
        pd.DataFrame(label_rows).drop_duplicates(),
        pd.DataFrame(resource_rows),
        pd.DataFrame(document_rows),
    )


WORK_ACTIVITY = {
    "software development": "Working with Computers",
    "infrastructure": "Working with Computers",
    "documentation": "Documenting/Recording Information",
    "procurement": "Monitoring and Controlling Resources",
    "contract management": "Evaluating Information to Determine Compliance with Standards",
    "budget": "Monitoring and Controlling Resources",
    "communication": "Communicating with Supervisors, Peers, or Subordinates",
    "policy": "Evaluating Information to Determine Compliance with Standards",
    "training": "Training and Teaching Others",
}

# Placeholder text containing the rule labeler's keywords, so the labeling stage
# can be tested end to end. Carries no real conversation content.
DOCUMENT_TEXT = {
    "software development": "Reviewing python code for a bug in this function and refactor the api repository.",
    "infrastructure": "Planning a server deployment to the cloud with a kubernetes pipeline across the network.",
    "documentation": "Please summarize this document into notes and a short report of the meeting minutes record.",
    "procurement": "Drafting a solicitation for a vendor bid and the acquisition purchase quote details.",
    "contract management": "Reviewing a contract clause and the modification award invoice deliverable schedule.",
    "budget": "Preparing a budget forecast for the cost and funding appropriation spend estimate.",
    "communication": "Drafting an email memo announcement and a newsletter message for internal draft review.",
    "policy": "Interpreting the policy regulation for compliance guidance under this directive and statute.",
    "training": "Building a training curriculum and onboarding tutorial course exercise for new staff.",
}


def main() -> None:
    conversations, labels, resources, documents = build()

    conversations.to_csv(HERE / "conversations.csv", index=False)
    labels.to_csv(HERE / "labels.csv", index=False)
    resources.to_csv(HERE / "resources.csv", index=False)
    documents.to_csv(HERE / "documents.csv", index=False)

    print(f"conversations: {len(conversations):,}")
    print(f"labels:        {len(labels):,}")
    print(f"resources:     {len(resources):,}")
    print(f"documents:     {len(documents):,}")
    print(f"users:         {conversations['user_id'].nunique():,}")


if __name__ == "__main__":
    main()
