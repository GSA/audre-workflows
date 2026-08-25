from __future__ import annotations

import pandas as pd
import pytest

from audre import privacy


def test_pseudonyms_are_stable_within_a_salt_and_differ_across_salts():
    values = pd.Series(["alice@example.test", "bob@example.test"])
    salt_a, salt_b = "salt-a", "salt-b"

    first = privacy.pseudonymize(values, salt=salt_a)
    again = privacy.pseudonymize(values, salt=salt_a)
    other = privacy.pseudonymize(values, salt=salt_b)

    assert first.tolist() == again.tolist(), "must be stable for longitudinal joins"
    assert first.tolist() != other.tolist(), "salt must actually key the mapping"
    assert not any(value in first.iloc[0] for value in ["alice", "example"])


def test_pseudonymize_requires_a_salt():
    with pytest.raises(ValueError, match="salt"):
        privacy.pseudonymize(pd.Series(["x"]), salt="")


def test_deidentify_drops_content_and_relabels_units():
    frame = pd.DataFrame(
        {
            "conversation_id": ["c1"],
            "user_id": ["someone@example.test"],
            "unit": ["Real Estate Office"],
            "content": ["sensitive prompt text"],
            "email": ["someone@example.test"],
        }
    )

    out = privacy.deidentify(
        frame, salt="s", unit_map={"Real Estate Office": "Unit B"}
    )

    assert "content" not in out.columns
    assert "email" not in out.columns
    assert out.loc[0, "unit"] == "Unit B"
    assert out.loc[0, "user_id"] != "someone@example.test"


def test_suppression_drops_cells_failing_either_threshold():
    frame = pd.DataFrame(
        {
            "label_value": ["big", "few_users", "few_convos"],
            "n_conversations": [100, 100, 2],
            "n_users": [50, 1, 50],
        }
    )

    kept = privacy.suppress_small_cells(frame)

    assert kept["label_value"].tolist() == ["big"]
    assert kept.attrs["n_suppressed"] == 2


def test_mask_mode_preserves_rows_but_hides_counts():
    frame = pd.DataFrame(
        {"label_value": ["big", "small"], "n_conversations": [100, 1], "n_users": [50, 1]}
    )

    masked = privacy.suppress_small_cells(
        frame, mode="mask", label_columns=("label_value",)
    )

    assert len(masked) == 2
    assert masked.loc[1, "label_value"] == privacy.SUPPRESSED
    assert pd.isna(masked.loc[1, "n_conversations"])


def test_assert_reportable_blocks_unsuppressed_output():
    frame = pd.DataFrame({"n_conversations": [100, 1], "n_users": [50, 1]})
    with pytest.raises(ValueError, match="reporting threshold"):
        privacy.assert_reportable(frame)


def test_assert_reportable_passes_after_suppression():
    frame = pd.DataFrame({"n_conversations": [100, 1], "n_users": [50, 1]})
    privacy.assert_reportable(privacy.suppress_small_cells(frame))


def test_missing_user_column_is_an_explicit_error_not_a_silent_skip():
    frame = pd.DataFrame({"n_conversations": [100]})
    with pytest.raises(KeyError, match="user_column=None"):
        privacy.suppress_small_cells(frame)
