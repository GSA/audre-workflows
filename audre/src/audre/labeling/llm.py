"""LLM-based conversation labeling against an OpenAI-compatible endpoint.

This is the labeler used to produce the case-study results the pipeline was
built for, generalized in three ways:

* the endpoint, model, and credentials come from arguments or environment
  variables rather than a vendored settings module, so it runs against any
  OpenAI-compatible chat-completions API;
* the closed-set value lists are rendered from a swappable taxonomy file
  instead of being pasted into the prompt;
* unparseable responses are preserved rather than dropped, so a labeling run
  can be repaired without re-spending tokens.

The labeler is off the reproduction path on purpose. Reviewers can reproduce
every result with :class:`audre.labeling.rules.RuleLabeler`, which requires no
credentials and no network access, while an organization runs this class
internally on data that cannot leave its boundary.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .base import LabelSet, Labeler, Taxonomy

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

DEFAULT_MAX_CONVERSATION_CHARACTERS = 50_000

#: Field definitions rendered into the taxonomy prompt. Keys must match the
#: field names in the taxonomy YAML.
FIELD_DEFINITIONS = {
    "work_activity": (
        "The single most fitting work activity describing the primary work "
        "behavior demonstrated in the conversation."
    ),
    "request_topic": (
        "The single most fitting topic describing the high-level domain of the "
        "user's request."
    ),
    "occupation_group": (
        "The single most fitting occupational group most associated with the "
        "conversation's subject matter. This describes the subject matter, not "
        "the user."
    ),
}

TITLE_EXAMPLES = (
    "Document Summarization, Contract Clause Identification, Procurement Strategy "
    "Insights, Records Request Aid, Supply Chain Analysis, Automated Compliance "
    "Reporting, Legislative Impact Analysis, Public Feedback Categorization, "
    "Testing Methodology, Deployment Strategy, Budget Breakdown, Market Analysis, "
    "Maintenance Plan"
)


def render_prompt(template_name: str, **context) -> str:
    """Render a bundled Jinja prompt template.

    ``StrictUndefined`` is used so that a renamed context variable fails loudly
    instead of quietly producing a prompt with a blank section, which would
    otherwise yield plausible-looking but meaningless labels.
    """

    environment = Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    return environment.get_template(template_name).render(**context)


def extract_json_dict(text: str) -> dict:
    """Extract a JSON object from a model response.

    Models wrap JSON in prose or code fences even when told not to, so parse
    the strict case first and fall back to the outermost brace-delimited span.
    """

    if not text:
        raise ValueError("empty response")

    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in response: {text[:200]!r}") from None
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


class LLMLabeler(Labeler):
    """Label conversations via an OpenAI-compatible chat-completions endpoint.

    Parameters
    ----------
    base_url, api_key:
        Endpoint and credential. Default to ``AUDRE_LLM_BASE_URL`` and
        ``AUDRE_LLM_API_KEY``. Credentials are never written to outputs; only
        the returned model name is recorded, as ``label_source``.
    model:
        Model identifier passed through to the endpoint.
    taxonomy:
        Closed-set taxonomy. Defaults to the bundled ``default.yaml``.
    kinds:
        Which label kinds to produce. Each closed-set kind costs one call, so
        request only what the analysis needs.
    temperature:
        Defaults to 0. Labels feed counts and shares, so run-to-run variation
        in labeling would be indistinguishable from a change in usage.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "llama_4_maverick",
        taxonomy: Taxonomy | None = None,
        kinds: tuple[str, ...] = ("topic_tag", "work_activity", "request_topic"),
        temperature: float = 0.0,
        max_conversation_characters: int = DEFAULT_MAX_CONVERSATION_CHARACTERS,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AUDRE_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("AUDRE_LLM_API_KEY", "")
        self.model = model
        self.taxonomy = taxonomy or Taxonomy.load()
        self.label_kinds = tuple(kinds)
        self.temperature = temperature
        self.max_conversation_characters = max_conversation_characters
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.timeout_seconds = timeout_seconds

        if not self.base_url or not self.api_key:
            raise ValueError(
                "LLMLabeler needs an endpoint and credential. Set AUDRE_LLM_BASE_URL "
                "and AUDRE_LLM_API_KEY, or pass base_url= and api_key=. To run the "
                "pipeline without any credential, use audre.labeling.rules.RuleLabeler."
            )

        self._taxonomy_kinds = tuple(
            kind for kind in self.label_kinds if kind in self.taxonomy.fields
        )

    # -- transport ---------------------------------------------------------

    def _complete(self, prompt_text: str) -> tuple[str, str]:
        """Return ``(content, model_name)``, retrying transient failures."""

        import requests  # imported lazily so the core package needs no HTTP stack

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content, data.get("model", self.model)
            except Exception as error:  # network, HTTP, or malformed envelope
                last_error = error
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))

        raise RuntimeError(f"chat completion failed after {self.max_retries} attempts") from last_error

    # -- prompts -----------------------------------------------------------

    def _taxonomy_prompt(self, conversation_text: str) -> str:
        return render_prompt(
            "taxonomy_classification.jinja",
            conversation_text=conversation_text,
            fields={kind: self.taxonomy.allowed(kind) for kind in self._taxonomy_kinds},
            field_definitions={
                kind: FIELD_DEFINITIONS.get(kind, f"The single most fitting {kind}.")
                for kind in self._taxonomy_kinds
            },
        )

    def _tags_prompt(self, conversation_text: str) -> str:
        return render_prompt(
            "topic_tags.jinja",
            conversation_text=conversation_text,
            min_broad=1,
            max_broad=3,
            min_specific=1,
            max_specific=3,
            min_messages=3,
        )

    def _title_prompt(self, conversation_text: str) -> str:
        return render_prompt(
            "use_case_title.jinja",
            conversation_text=conversation_text,
            examples=TITLE_EXAMPLES,
        )

    # -- labeling ----------------------------------------------------------

    def label_one(self, conversation_id: str, text: str) -> LabelSet:
        conversation_text = (text or "")[: self.max_conversation_characters]
        result = LabelSet(conversation_id=conversation_id, label_source=f"llm:{self.model}")

        if not conversation_text.strip():
            result.parse_success = False
            result.raw_response = ""
            return result

        raw_parts: list[str] = []
        failed = False

        if self._taxonomy_kinds:
            content, model_name = self._complete(self._taxonomy_prompt(conversation_text))
            result.label_source = f"llm:{model_name}"
            raw_parts.append(content)
            try:
                parsed = extract_json_dict(content)
            except ValueError:
                failed = True
            else:
                for kind in self._taxonomy_kinds:
                    value = parsed.get(kind)
                    # Off-taxonomy values are kept, not coerced. Silently
                    # snapping them to a legal value would hide the model's
                    # failure rate; Taxonomy.coverage() surfaces them instead.
                    result.values[kind] = value if value else None

        if "topic_tag" in self.label_kinds:
            content, _ = self._complete(self._tags_prompt(conversation_text))
            raw_parts.append(content)
            try:
                parsed = extract_json_dict(content)
                tags = parsed.get("tags") or []
                result.values["topic_tag"] = [
                    str(tag).strip().lower() for tag in tags if str(tag).strip()
                ]
            except ValueError:
                failed = True

        if "use_case_title" in self.label_kinds:
            content, _ = self._complete(self._title_prompt(conversation_text))
            raw_parts.append(content)
            try:
                result.values["use_case_title"] = extract_json_dict(content).get("use_case")
            except ValueError:
                failed = True

        if failed:
            result.parse_success = False
            result.raw_response = "\n---\n".join(raw_parts)
        return result
