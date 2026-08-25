# Test fixtures

A small hand-written dataset used to smoke-test pipeline wiring: that stages
compose, that suppression fires, and that edge cases (single-conversation users,
multi-tag conversations, missing cost records, conversations outside the
transition window) are handled.

**This is not a simulation study.** It is far too small and too regular to
support any claim about the pipeline's ability to recover planted patterns, and
no result computed from it should be reported. Statistical validation —
synthetic-pattern recovery, window sensitivity, and missingness sensitivity —
is a separate piece of work.

Files follow the AUDRE schema exactly (see `docs/SCHEMA.md`):

- `conversations.csv` — the analysis spine, with no conversation content
- `labels.csv` — long format, one row per (conversation, kind, value)
- `resources.csv` — token and cost measures, deliberately incomplete so that
  observed-versus-estimated provenance handling is exercised
- `documents.csv` — synthetic placeholder text, used only to test the labeling
  stage; carries no real content
