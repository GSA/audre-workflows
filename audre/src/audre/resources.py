"""Resource-intensity measures: tokens and estimated cost.

The "R" layer of AUDRE. Frequently observed activities are not uniformly the
most resource-intensive, so resource use is treated as a view in its own right
rather than a proxy for importance.

Two honesty constraints are enforced structurally:

* Estimated values are never silently mixed with observed ones. Every row
  carries a ``cost_source`` provenance flag, and summaries report the two
  populations separately as well as combined.
* Character-based token estimation is offered, but flagged. It is an
  approximation of transcript-level inference cost, not a billing record, and
  it excludes every other program expense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import privacy

#: Default characters-per-token divisor. Matches the coarse heuristic used in
#: the original case study. Real tokenizers differ substantially by model and
#: by language, so prefer :func:`estimate_tokens` with an explicit
#: ``chars_per_token`` calibrated against the tokenizer you actually use.
CHARS_PER_TOKEN = 4.0

OBSERVED = "observed"
ESTIMATED = "estimated"
IMPUTED = "median_imputed"


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for one model.

    Prices are supplied by the caller rather than bundled, because published
    prices change and a stale hardcoded table silently corrupts every cost
    figure downstream.
    """

    model: str
    prompt_usd_per_mtok: float
    completion_usd_per_mtok: float

    def cost(self, prompt_tokens: float, completion_tokens: float) -> float:
        return (
            prompt_tokens * self.prompt_usd_per_mtok
            + completion_tokens * self.completion_usd_per_mtok
        ) / 1_000_000


def price_table(prices: dict[str, tuple[float, float]]) -> dict[str, ModelPrice]:
    """Build a price lookup from ``{model: (prompt_rate, completion_rate)}``."""

    return {
        model: ModelPrice(model, prompt_rate, completion_rate)
        for model, (prompt_rate, completion_rate) in prices.items()
    }


def estimate_tokens(
    character_counts: pd.Series, *, chars_per_token: float = CHARS_PER_TOKEN
) -> pd.Series:
    """Estimate token counts from character counts.

    A deliberately crude heuristic, exposed so that the assumption is visible
    and can be varied in a sensitivity analysis rather than buried in a script.
    """

    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return (pd.to_numeric(character_counts, errors="coerce") / chars_per_token).round()


def estimate_cost(
    resources: pd.DataFrame,
    conversations: pd.DataFrame,
    prices: dict[str, ModelPrice],
    *,
    model_column: str = "model",
    default_price: ModelPrice | None = None,
) -> pd.DataFrame:
    """Add ``estimated_cost_usd`` from token counts and a price table.

    Conversations whose model has no price and no ``default_price`` receive a
    null estimate rather than zero, so that unpriced activity is visible as
    missing instead of masquerading as free.
    """

    required = {"prompt_tokens", "completion_tokens"}
    missing = required - set(resources.columns)
    if missing:
        raise KeyError(f"resources is missing token column(s): {sorted(missing)}")

    out = resources.merge(
        conversations[["conversation_id", model_column]], on="conversation_id", how="left"
    )

    def row_cost(row: pd.Series) -> float:
        price = prices.get(row[model_column], default_price)
        if price is None or pd.isna(row["prompt_tokens"]) or pd.isna(row["completion_tokens"]):
            return np.nan
        return price.cost(row["prompt_tokens"], row["completion_tokens"])

    out["estimated_cost_usd"] = out.apply(row_cost, axis=1)
    out["priced_model"] = out[model_column].isin(prices) | (default_price is not None)
    return out


def resolve_cost(
    resources: pd.DataFrame,
    *,
    observed_column: str = "cost_usd",
    estimated_column: str = "estimated_cost_usd",
    impute_median: bool = False,
) -> pd.DataFrame:
    """Produce a single ``analysis_cost_usd`` column with explicit provenance.

    Precedence is observed, then estimated, then (optionally) the median of the
    observed distribution.

    Median imputation is off by default. It compresses variance and biases
    per-conversation comparisons toward the centre, which is precisely the
    quantity a resource-intensity analysis is trying to measure. Enable it only
    for totals, and always report the observed/imputed split alongside.
    """

    out = resources.copy()
    observed = (
        pd.to_numeric(out[observed_column], errors="coerce")
        if observed_column in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    estimated = (
        pd.to_numeric(out[estimated_column], errors="coerce")
        if estimated_column in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    cost = observed.copy()
    source = pd.Series(pd.NA, index=out.index, dtype="string")
    source[observed.notna()] = OBSERVED

    use_estimated = cost.isna() & estimated.notna()
    cost[use_estimated] = estimated[use_estimated]
    source[use_estimated] = ESTIMATED

    if impute_median and observed.notna().any():
        median_observed = float(observed.median())
        use_imputed = cost.isna()
        cost[use_imputed] = median_observed
        source[use_imputed] = IMPUTED
        out.attrs["imputation_median_usd"] = median_observed

    out["analysis_cost_usd"] = cost
    out["cost_source"] = source
    out["has_observed_cost"] = observed.notna()
    return out


def cost_coverage(resources: pd.DataFrame) -> pd.DataFrame:
    """Report how much of the cost signal is observed versus derived.

    Coverage belongs next to every cost figure. A cost total built from 20%
    observed records supports very different claims from one built from 95%.
    """

    if "cost_source" not in resources.columns:
        raise KeyError("call resolve_cost() before cost_coverage()")

    out = (
        resources.groupby("cost_source", dropna=False)
        .agg(
            n_conversations=("conversation_id", "nunique"),
            total_cost_usd=("analysis_cost_usd", "sum"),
            mean_cost_usd=("analysis_cost_usd", "mean"),
            median_cost_usd=("analysis_cost_usd", "median"),
        )
        .reset_index()
    )
    out["share_of_conversations"] = out["n_conversations"] / out["n_conversations"].sum()
    return out


def resource_intensity_by_label(
    resources: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_kind: str = "work_activity",
    conversations: pd.DataFrame | None = None,
    suppress: bool = True,
) -> pd.DataFrame:
    """Average tokens and cost per conversation, per label.

    Pairing per-conversation intensity with conversation volume is the point of
    this view: it separates activities that are common from activities that are
    expensive, which are frequently not the same set.
    """

    subset = labels.loc[labels["label_kind"] == label_kind].drop_duplicates(
        subset=["conversation_id", "label_value"]
    )
    if subset.empty:
        raise ValueError(f"No labels of kind {label_kind!r}")

    merged = subset.merge(resources, on="conversation_id", how="inner")
    if conversations is not None:
        merged = merged.merge(
            conversations[["conversation_id", "user_id"]], on="conversation_id", how="left"
        )

    aggregations: dict[str, tuple[str, str]] = {
        "n_conversations": ("conversation_id", "nunique"),
    }
    if "user_id" in merged.columns:
        aggregations["n_users"] = ("user_id", "nunique")
    for column, alias in (
        ("total_tokens", "tokens"),
        ("analysis_cost_usd", "cost_usd"),
    ):
        if column in merged.columns:
            aggregations[f"mean_{alias}"] = (column, "mean")
            aggregations[f"median_{alias}"] = (column, "median")
            aggregations[f"total_{alias}"] = (column, "sum")

    if "has_observed_cost" in merged.columns:
        aggregations["observed_cost_coverage"] = ("has_observed_cost", "mean")

    out = (
        merged.groupby("label_value", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("n_conversations", ascending=False, ignore_index=True)
    )

    if not suppress:
        return out
    user_column = "n_users" if "n_users" in out.columns else None
    return privacy.suppress_small_cells(out, user_column=user_column)
