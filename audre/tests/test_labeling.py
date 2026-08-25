from __future__ import annotations

import pandas as pd
import pytest

from audre import schema
from audre.labeling import RuleLabeler, Taxonomy
from audre.labeling.llm import extract_json_dict, render_prompt


# -- taxonomy --------------------------------------------------------------


def test_bundled_taxonomy_loads_with_expected_fields():
    taxonomy = Taxonomy.load()
    assert {"work_activity", "request_topic", "occupation_group"} <= set(taxonomy.fields)
    assert taxonomy.is_valid("work_activity", "Working with Computers")
    assert not taxonomy.is_valid("work_activity", "Inventing New Categories")


def test_neutral_taxonomy_is_structurally_interchangeable():
    """The neutral taxonomy must be a drop-in swap for the default.

    It exists so that a blind-review or non-US deployment can change one
    argument rather than rewrite prompts, which only holds if both files expose
    the same field names.
    """

    default = Taxonomy.load("default.yaml")
    neutral = Taxonomy.load("neutral.yaml")

    assert set(default.fields) == set(neutral.fields)
    assert all(len(values) > 0 for values in neutral.fields.values())


def test_neutral_taxonomy_values_are_jurisdiction_neutral():
    """Only the category values are checked, not the prose.

    The point of this file is that the labels written into output tables carry no
    national-classification wording; the description is free to explain why.
    """

    neutral = Taxonomy.load("neutral.yaml")
    values = " ".join(
        value for values in neutral.fields.values() for value in values
    ).lower()

    for term in ("o*net", "onet", "standard occupational", "u.s."):
        assert term not in values


def test_unknown_taxonomy_field_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        Taxonomy.load().allowed("not_a_field")


def test_missing_taxonomy_file_lists_bundled_options():
    with pytest.raises(FileNotFoundError, match="Bundled"):
        Taxonomy.load("does_not_exist.yaml")


def test_taxonomy_coverage_flags_off_taxonomy_values():
    labels = pd.DataFrame(
        {
            "conversation_id": ["c1", "c2"],
            "label_kind": ["work_activity"] * 2,
            "label_value": ["Working with Computers", "Something Invented"],
        }
    )
    coverage = Taxonomy.load().coverage(labels)
    invented = coverage[coverage["label_value"] == "Something Invented"]

    assert bool(invented["off_taxonomy"].iloc[0]) is True
    used = coverage[coverage["label_value"] == "Working with Computers"]
    assert int(used["n_conversations"].iloc[0]) == 1
    assert (coverage["n_conversations"] == 0).any(), "unused categories are reported too"


# -- rule labeler ----------------------------------------------------------


def test_rule_labeler_is_deterministic(documents):
    labeler = RuleLabeler()
    first = labeler.label(documents.head(20))
    second = labeler.label(documents.head(20))

    pd.testing.assert_frame_equal(first, second)


def test_rule_labeler_output_validates_against_the_labels_schema(documents):
    labels = RuleLabeler().label(documents)
    report = schema.validate(schema.coerce(labels, schema.LABELS), "labels")

    assert report.ok, str(report)


def test_rule_labeler_assigns_taxonomy_values_from_the_taxonomy(documents):
    taxonomy = Taxonomy.load()
    labels = RuleLabeler().label(documents)
    assigned = labels[labels["label_kind"] == "work_activity"]["label_value"].unique()

    assert len(assigned) > 0
    assert all(taxonomy.is_valid("work_activity", value) for value in assigned)


def test_rule_labeler_stamps_a_source(documents):
    labels = RuleLabeler().label(documents.head(5))
    assert (labels["label_source"] == "rules:keyword-v1").all()


def test_taxonomy_kinds_stay_single_valued(documents):
    labels = RuleLabeler().label(documents)
    per_conversation = (
        labels[labels["label_kind"] == "work_activity"]
        .groupby("conversation_id")
        .size()
    )
    assert (per_conversation == 1).all()


def test_topic_tags_may_be_multi_valued(documents):
    labels = RuleLabeler().label(documents)
    per_conversation = (
        labels[labels["label_kind"] == "topic_tag"].groupby("conversation_id").size()
    )
    assert per_conversation.max() >= 1
    assert per_conversation.max() <= 3


def test_unmatched_text_falls_back_rather_than_failing():
    documents = pd.DataFrame(
        {"conversation_id": ["c1"], "document": ["zzz qqq unintelligible"]}
    )
    labels = RuleLabeler().label(documents)
    tags = labels[labels["label_kind"] == "topic_tag"]["label_value"].tolist()

    assert tags == ["general"]


def test_missing_text_column_raises(documents):
    with pytest.raises(KeyError, match="missing"):
        RuleLabeler().label(documents, text_column="nope")


# -- prompt rendering and parsing -----------------------------------------


def test_taxonomy_prompt_renders_the_loaded_taxonomy_not_a_hardcoded_list():
    taxonomy = Taxonomy.load()
    prompt = render_prompt(
        "taxonomy_classification.jinja",
        conversation_text="example",
        fields={"work_activity": taxonomy.allowed("work_activity")},
        field_definitions={"work_activity": "the work activity"},
    )

    assert "Working with Computers" in prompt
    assert "example" in prompt
    # The prompt must not leak an identity or coach an occupational inference.
    assert "Do NOT infer the user's job" in prompt


def test_prompt_rendering_fails_loudly_on_a_missing_variable():
    from jinja2 import UndefinedError

    with pytest.raises(UndefinedError):
        render_prompt("topic_tags.jinja", conversation_text="x")


@pytest.mark.parametrize(
    "response",
    [
        '{"tags": ["a"]}',
        '```json\n{"tags": ["a"]}\n```',
        'Possible JSON response: {"tags": ["a"]}',
        '{"tags": ["a"]}\n\nHope that helps!',
    ],
)
def test_extract_json_dict_survives_common_model_wrappers(response):
    assert extract_json_dict(response) == {"tags": ["a"]}


@pytest.mark.parametrize("response", ["", "no json here", "[1, 2, 3]"])
def test_extract_json_dict_rejects_unusable_responses(response):
    with pytest.raises(ValueError):
        extract_json_dict(response)


def test_llm_labeler_refuses_to_construct_without_credentials(monkeypatch):
    from audre.labeling import LLMLabeler

    monkeypatch.delenv("AUDRE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AUDRE_LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="RuleLabeler"):
        LLMLabeler()
