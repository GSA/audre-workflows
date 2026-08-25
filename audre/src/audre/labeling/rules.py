"""Deterministic keyword labeler.

Exists so that the whole pipeline — including the validation harness — runs with
no credentials, no network access, and no run-to-run variation. That makes every
published number in the methods paper reproducible by a reader, which an
LLM-only labeling path cannot offer.

This labeler is a reproducibility instrument, not a replacement for LLM
labeling. It matches surface keywords and will therefore miss paraphrase,
implication, and anything the vocabulary does not anticipate. Do not report
substantive findings from it on real data.
"""

from __future__ import annotations

import re
from collections import Counter

from .base import LabelSet, Labeler, Taxonomy

#: Keyword vocabulary per topic tag. Deliberately small and legible: this is a
#: fixture for reproducible tests, so it must be auditable at a glance.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "software development": ("code", "python", "function", "bug", "refactor", "api", "repository"),
    "data analysis": ("dataset", "analysis", "statistics", "regression", "chart", "aggregate"),
    "cybersecurity": ("vulnerability", "threat", "phishing", "encryption", "incident", "patch"),
    "infrastructure": ("server", "deployment", "cloud", "kubernetes", "pipeline", "network"),
    "procurement": ("solicitation", "vendor", "bid", "acquisition", "purchase", "quote"),
    "contract management": ("contract", "clause", "modification", "award", "invoice", "deliverable"),
    "facilities": ("building", "lease", "space", "maintenance", "hvac", "occupancy"),
    "project management": ("milestone", "schedule", "timeline", "deliverable", "sprint", "status"),
    "communication": ("email", "memo", "announcement", "draft", "message", "newsletter"),
    "policy": ("policy", "regulation", "compliance", "guidance", "directive", "statute"),
    "training": ("training", "onboarding", "curriculum", "tutorial", "course", "exercise"),
    "budget": ("budget", "cost", "funding", "appropriation", "forecast", "spend"),
    "research": ("literature", "study", "evidence", "survey", "findings", "hypothesis"),
    "documentation": ("document", "summarize", "notes", "report", "minutes", "record"),
}

#: Mapping from topic tag to closed-set taxonomy values, so the rule labeler can
#: populate the same label kinds as the LLM labeler.
TOPIC_TO_TAXONOMY: dict[str, dict[str, str]] = {
    "software development": {
        "work_activity": "Working with Computers",
        "request_topic": "Software Development",
        "occupation_group": "Computer and Mathematical",
    },
    "data analysis": {
        "work_activity": "Analyzing Data or Information",
        "request_topic": "Data Analysis & Business Intelligence",
        "occupation_group": "Computer and Mathematical",
    },
    "cybersecurity": {
        "work_activity": "Monitoring Processes, Materials, or Surroundings",
        "request_topic": "Cybersecurity & Threat Detection",
        "occupation_group": "Computer and Mathematical",
    },
    "infrastructure": {
        "work_activity": "Working with Computers",
        "request_topic": "DevOps & Infrastructure Operations",
        "occupation_group": "Computer and Mathematical",
    },
    "procurement": {
        "work_activity": "Monitoring and Controlling Resources",
        "request_topic": "Business Process & Operations",
        "occupation_group": "Business and Financial Operations",
    },
    "contract management": {
        "work_activity": "Evaluating Information to Determine Compliance with Standards",
        "request_topic": "Compliance & Regulatory",
        "occupation_group": "Business and Financial Operations",
    },
    "facilities": {
        "work_activity": "Inspecting Equipment, Structures, or Materials",
        "request_topic": "Business Process & Operations",
        "occupation_group": "Installation, Maintenance, and Repair",
    },
    "project management": {
        "work_activity": "Organizing, Planning, and Prioritizing Work",
        "request_topic": "Business Process & Operations",
        "occupation_group": "Management",
    },
    "communication": {
        "work_activity": "Communicating with Supervisors, Peers, or Subordinates",
        "request_topic": "Content Creation & Copywriting",
        "occupation_group": "Office and Administrative Support",
    },
    "policy": {
        "work_activity": "Evaluating Information to Determine Compliance with Standards",
        "request_topic": "Compliance & Regulatory",
        "occupation_group": "Legal",
    },
    "training": {
        "work_activity": "Training and Teaching Others",
        "request_topic": "Education & Learning",
        "occupation_group": "Educational Instruction and Library",
    },
    "budget": {
        "work_activity": "Monitoring and Controlling Resources",
        "request_topic": "Business Process & Operations",
        "occupation_group": "Business and Financial Operations",
    },
    "research": {
        "work_activity": "Getting Information",
        "request_topic": "Research & Intelligence",
        "occupation_group": "Life, Physical, and Social Science",
    },
    "documentation": {
        "work_activity": "Documenting/Recording Information",
        "request_topic": "Document Processing & Extraction",
        "occupation_group": "Office and Administrative Support",
    },
}

FALLBACK_TOPIC = "general"
FALLBACK_TAXONOMY = {
    "work_activity": "Getting Information",
    "request_topic": "Other / Unclear",
    "occupation_group": "Office and Administrative Support",
}

_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_\-]+")


class RuleLabeler(Labeler):
    """Assign topic tags and taxonomy values by keyword matching.

    Parameters
    ----------
    max_tags:
        Maximum topic tags per conversation. Multi-tagging is retained because
        it is a real property of the data that the transition and co-occurrence
        layers must handle.
    min_hits:
        Minimum keyword matches before a topic is assigned. Raising it trades
        recall for precision.
    """

    def __init__(
        self,
        *,
        taxonomy: Taxonomy | None = None,
        kinds: tuple[str, ...] = ("topic_tag", "work_activity", "request_topic"),
        max_tags: int = 3,
        min_hits: int = 1,
        keywords: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.taxonomy = taxonomy or Taxonomy.load()
        self.label_kinds = tuple(kinds)
        self.max_tags = max_tags
        self.min_hits = min_hits
        self.keywords = keywords or TOPIC_KEYWORDS

    def label_one(self, conversation_id: str, text: str) -> LabelSet:
        result = LabelSet(conversation_id=conversation_id, label_source="rules:keyword-v1")
        tokens = set(_TOKEN_PATTERN.findall((text or "").lower()))

        hits = Counter(
            {
                topic: sum(1 for keyword in keywords if keyword in tokens)
                for topic, keywords in self.keywords.items()
            }
        )
        ranked = [
            topic
            for topic, count in sorted(hits.items(), key=lambda item: (-item[1], item[0]))
            if count >= self.min_hits
        ][: self.max_tags]

        if not ranked:
            ranked = [FALLBACK_TOPIC]

        if "topic_tag" in self.label_kinds:
            result.values["topic_tag"] = ranked

        # Closed-set kinds derive from the highest-ranked topic, keeping them
        # single-valued and therefore countable as shares.
        primary = ranked[0]
        mapping = TOPIC_TO_TAXONOMY.get(primary, FALLBACK_TAXONOMY)
        for kind in self.label_kinds:
            if kind in self.taxonomy.fields:
                result.values[kind] = mapping.get(kind, FALLBACK_TAXONOMY.get(kind))

        return result
