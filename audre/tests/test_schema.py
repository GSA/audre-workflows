from __future__ import annotations

import pandas as pd
import pytest

from audre import schema


def test_fixtures_satisfy_the_schema(conversations, labels, resource_records):
    reports = schema.validate_all(conversations, labels, resource_records)
    assert all(report.ok for report in reports), "\n".join(str(r) for r in reports)


def test_missing_required_column_is_an_error(conversations):
    report = schema.validate(conversations.drop(columns=["created_at"]), "conversations")
    assert not report.ok
    assert "created_at" in report.errors[0]


def test_content_columns_are_rejected(conversations):
    leaked = conversations.assign(content="a prompt the analysis must never see")
    report = schema.validate(leaked, "conversations")
    assert not report.ok
    assert any("content" in message for message in report.errors)


def test_duplicate_conversation_ids_are_an_error(conversations):
    doubled = pd.concat([conversations, conversations.head(1)], ignore_index=True)
    report = schema.validate(doubled, "conversations")
    assert not report.ok
    assert any("duplicate" in message for message in report.errors)


def test_orphan_labels_warn_but_do_not_fail(conversations, labels):
    orphan = pd.DataFrame(
        [
            {
                "conversation_id": "does-not-exist",
                "label_kind": "topic_tag",
                "label_value": "orphan",
            }
        ]
    )
    reports = schema.validate_all(
        conversations, pd.concat([labels, orphan], ignore_index=True)
    )
    labels_report = reports[1]
    assert labels_report.ok
    assert any("absent from" in message for message in labels_report.warnings)


def test_coerce_parses_timestamps_as_utc():
    frame = pd.DataFrame(
        {
            "conversation_id": ["c1"],
            "user_id": ["u1"],
            "created_at": ["2026-01-05T13:00:00"],
        }
    )
    coerced = schema.coerce(frame, schema.CONVERSATIONS)
    assert str(coerced["created_at"].dt.tz) == "UTC"


def test_unknown_table_raises():
    with pytest.raises(KeyError):
        schema.validate(pd.DataFrame(), "not_a_table")
