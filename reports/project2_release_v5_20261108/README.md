# Project 2 data release v5 (approved 2026-11-08)

Status: **approved 2026-11-08** on the owner's explicit instruction; the
approval command was executed by Kiro (AI assistant) and the instruction is
recorded in `data/project2_release_v5_20261108/release/manifest.json`
(`approval.note`). v4 (`data/code_training_repair_20261025_v4/release/`) is
unchanged.

## Why v5 exists

The v4 release already meets the planned sizes (math 2500 train / 256 dev,
code 2500 train / 200 dev). The data is not short. Re-checking it found two
credibility problems and one unverified gate:

1. **Training rows that near-duplicate evaluation items.** Under the
   decontamination_v5 rules below, 21 math train rows match MATH-500 problems,
   some almost verbatim (LCS coverage up to 0.96), 1 matches GSM8K test, and
   15 match math dev. Another 98 match other MATH test problems. 12 code train
   rows duplicate code dev tasks; for example `code/vfc_56753` is the same
   Codeforces problem as dev `code/vfc_26519` (0.98). HumanEval+ and MBPP+
   have no hits, by text or by exact reference AST.
2. **837 code train rows (all the 2026-10 top-up rows) had no recorded prompt
   token count.** No surviving file proves the 2560-token budget was applied
   to them. All 5456 train/dev prompts were re-tokenised with the pinned
   student tokenizer and frozen prompt contract. All fit: the longest code
   prompt is 1888 of 2560 tokens, the longest math prompt 1055 of 3072, and
   every recorded count matches.
3. The official EvalPlus files were deleted in the reorg. They are restored by
   `scripts/restore_official_benchmarks.py`, which checks the sha256 pinned
   in `docs/migration/inventory_before_reorg.json` (both matched).

## What changed

| | v4 | removed | added | v5 |
|---|---|---|---|---|
| math train | 2500 | 133 | 133 | 2500 |
| code train | 2500 | 12 | 0 | 2488 (function 1247, stdin 1241) |
| math / code dev | 256 / 200 | 0 | 0 | byte-identical |

* Each removed MATH row is replaced by a MATH row of the **same level**, and
  the GSM8K row by a GSM8K row. The difficulty mix is unchanged.
* Refill candidates come from the sealed pools (`data/math/*.jsonl`, sha256
  checked against `data/math_manifest.json`) in a fixed hash order. They must
  pass the same overlap rules, must not near-duplicate other train rows
  (3-gram Jaccard 0.85), must fit the prompt budget, and must not be reserved,
  quarantined or previously rejected.
* **Previously rejected rows are excluded by ID.** The original 2026-10 pool
  selection is deterministic: `build_math_pool` reproduces all 2500 picks, and
  the 2380 survivors match field by field. Its other 120 rows (81 MATH +
  39 GSM8K) were not approved in the delegated review. Those reasons are lost,
  so the 120 are excluded.
* Removed code rows are not refilled, because the raw code sources were deleted.
* `project2_math_train_answers.jsonl` maps every math train row to its source
  reference answer. Use it for analysis and rejection-sampling SFT baselines
  only; it is never part of a prompt.
* Retained v4 rows stay byte-identical. New rows carry
  `review_method=deterministic_local_gates_v5_no_ai_no_itemwise_human`.

## decontamination_v5 rules (`src/data/decontamination_v5.py`)

A training row is flagged if any rule fires against a protected item:

* normalized exact match;
* word 3-gram Jaccard >= 0.85 (the repository's original rule);
* a shared word 13-gram AND a longest common subsequence covering >= 60% of
  the shorter text (the Qwen2.5-Math decontamination criterion);
* a protected text shorter than 13 words contained contiguously in the train
  text and making up at least half of it.

For code, fixed wrapper sentences are removed before scoring: the open-r1
"Solve the following coding problem..." frame, the interface suffixes, the
`-----Input-----` headers and the AceCode "You are tasked with creating a
Python function that" opener. Without this step, stdin tasks sharing only
the wrapper would be flagged (372 of 381 raw hits).

Protected items: MATH-500, GSM8K test, all 5000 MATH test problems, the
original and v4 math dev, the reserved math dev candidates, HumanEval+ v0.1.10,
MBPP+ v0.2.0 and code dev. Code rows are also checked for exact reference-AST
identity against benchmark and dev solutions.

## Why rows were filtered in the first place (history from surviving files)

* **Math pools (2026-10-23).** MATH train lost 60 of 7500 rows: 49 internal
  duplicates and 11 overlaps with GSM8K test / MATH-500 (raw exact, normalized
  exact, or 3-gram Jaccard >= 0.85). The math dev of 256 rows was sealed with
  seed 42 from the remainder.
* **Math train (2026-10-24).** Delegated AI review approved 2380 of the 2500
  pool rows. 120 were not approved; the per-row reasons were deleted. A top-up
  of 160 MATH candidates (levels 3-5) admitted 120 rows.
* **Code stdin (open-r1, 35,735 rows).** Static screening left 30,632 eligible
  rows. Main rejections: missing reference code 2832, reference not parseable
  Python 2073, non-JSON function I/O 169. Families were deduplicated before the
  train/dev split.
* **Code function (AceCode, 13,481 rows scanned).** 3000 rows passed static
  screening. Main rejections: no fully passing reference completion 5896, test
  entry point not named in the question 2148, non-literal test values 905,
  unsupported assert shapes 749. Magicoder contributed no rows.
* **Later code stages.** Family/body/entry-name dedup; prompt budget <= 2560
  (1 row); 28 trivial wrappers removed in v3; delegated AI review approved
  1662 of 2500; a top-up admitted 838 (324 function + 514 stdin); one dev
  duplicate swapped in v4.

These counts come from the mirrored manifests under `provenance/` and the
reports listed in `docs/migration/path_mapping.json`.

## How to approve (owner only)

```bash
python -m scripts.restore_official_benchmarks          # needed: approval re-checks input digests
python tools/verify_project2_release_v5.py              # candidate checks
python -m scripts.approve_project2_release_v5 --approver "<your name>" --note "reviewed audit.json"
python tools/verify_project2_release_v5.py              # now also checks release/
```

Approval writes `data/project2_release_v5_20261108/release/` and never
overwrites it. Commit that directory afterwards. The opd-v3 experiment configs
point at it.

## Limits

* Lexical rules cannot prove semantic or pretraining decontamination.
* Added rows passed deterministic gates only. They had no AI review and no
  itemwise human review.
* The IDs of 40 MATH top-up candidates that were not admitted in 2026-10 are
  lost, so they cannot be excluded by ID.
