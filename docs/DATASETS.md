# Datasets (verified 2026-11-21)

Raw data lives in `data/raw/` (gitignored). `python scripts/download_data.py` fetches every
dataset at a pinned revision and writes `data/raw/raw_manifest.json` with checksums.

## PrefEval: main benchmark
- **Source:** github.com/amazon-science/PrefEval, commit `50795054b5ff5f418d2b768a331d71e480f93331`. Paper: Zhao et al., ICLR 2025, arXiv:2502.09597.
- **Licence:** **CC BY-NC 4.0** (non-commercial). Raw files are not redistributed here. Only the 4 judge prompts are copied, verbatim with attribution, in `src/scopemem/prompts/judge_prefeval/`.
- **Layout:**
  - `benchmark_dataset/explicit_preference/<topic>.json`: a list of `{preference, question, explanation}`. **20 topics, 1,000 pairs**, 31 (education_learning_styles) to 62 (entertain_shows) per topic. 988 unique preference texts.
  - `benchmark_dataset/implicit_preference/choice-based/<topic>.json`: the same pairs, plus `implicit_query`, `options[4]`, `conversation{query, assistant_options, user_selection, assistant_acknowledgment}` and `aligned_op`.
  - `benchmark_dataset/implicit_preference/persona-driven/<topic>.json`: the same pairs, plus `persona` and `conversation{"0": {user, assistant}, ...}`.
  - `benchmark_dataset/filtered_inter_turns.json`: 24 filler conversations (LMSYS-style, keys `conversation_id, model, conversation[{role, content}], turn, language, redacted`). This is 316 user/assistant exchanges.
  - `benchmark_dataset/mcq_options/`, `rag_retrieval/`: not used.
- **Judge protocol** (`generation_task/llm_based_evaluation_errortypes.py`, `get_preference_following_accuracy_generation_task.py`):
  - The judge runs 4 checks, each with XML output: `acknow`, `violate`, `hallucinate` (compares the preference that `acknow` extracted against the original) and `helpful`. Temperature is 0.
  - The decision logic is quoted in `src/scopemem/eval/judge.py`.
  - **Preference following = NOT (inconsistent OR hallucinated-preference violation OR preference-unaware violation OR unhelpful).** This is PrefEval's combined criterion: a response counts as following only if it is helpful and has none of the violation types.
- **Generation protocol in PrefEval** (for reference): the preference is a user turn; the model's own reply to it is kept; filler turns follow; the question gets the suffix "(Please respond within {max_tokens} words.)"; the system prompt is "You are a helpful assistant."; max_gen_tokens is 300. Our differences are listed in DECISIONS D-011.

## PrefEval-Multi (built here)
- Builder: `python -m scopemem.data.build_multi` (`make data`). Output goes to `data/processed/`: `prefeval_multi_N{n}.jsonl` (full) and `prefeval_multi_N{n}_sub500.jsonl` (E3), plus `prefs.jsonl`, `folds.json` and `manifest.json` (SHA-256 per file).
- Summary statistics (seed 13):

| set | N | instances | positive | null | null share | with global | mean \|G\| (pos) | history segments |
|---|---|---|---|---|---|---|---|---|
| full | 1 | 1250 | 1000 | 250 | 0.200 | 0 | 1.00 | 4 |
| full | 5 / 10 / 25 / 50 / 100 | 1250 | 1000 | 250 | 0.200 | 298 | 1.30 | 20 / 40 / 100 / 200 / 400 |
| sub500 | 1 | 625 | 500 | 125 | 0.200 | 0 | 1.00 | 4 |
| sub500 | 5 … 100 | 625 | 500 | 125 | 0.200 | 143 | 1.29 | 20 … 400 |

- **Instances per fold** (full): 257 / 246 / 237 / 240 / 270. There are 5 folds of 4 topics each; an instance's fold is the fold of its query topic.

### How easy is PrefEval-Multi for similarity retrieval? (measured 2026-11-21)
`bge-base-en-v1.5`, full set, N=25, 1,000 positives, cosine(query, preference text):

| recall@1 | recall@3 | recall@5 | recall@10 | median rank of target | target − best distractor (quartiles) |
|---|---|---|---|---|---|
| 0.696 | 0.909 | 0.960 | 0.991 | 1 | −0.012 / 0.044 / 0.089 |

If the distractors came from the target's own topic, recall@1 would be 0.099 and recall@5 0.351. The hardest topics (recall@5 ≈ 0.90) are travel_activities, lifestyle_health and pet_ownership. **This is why the cross-topic split was added (DECISIONS D-033).**

## Cross-topic split `xtopic` (built here, D-033)
- **Pipeline:** `python -m scopemem.data.cross_topic stage=generate|verify|select|export_audit|score_audit`. Outputs go to `data/processed/cross_topic/`:
  - `candidates__<generator>.jsonl` and `verified__<generator>__<verifier>.jsonl`;
  - `xtopic_items.jsonl` (the `primary` and `reserve` roles) and `select_stats.json`.
- **Instance files:** `prefeval_multi_N{n}_xtopic.jsonl` and `prefeval_multi_N{n}_xtopic_sub500.jsonl`.
- **Size:** at most one primary request per unique preference, so at most 988 positives plus 20% nulls. The final count depends on the verifier pass rate, which will be recorded here after the cluster run.
- **Human audit:** a blind CSV of about 150 rows. Kept-request precision and verifier κ go to `results/tables/xtopic_audit.json`.

## PersonaMem (E8)
- **Source:** HF dataset `bowen-upenn/PersonaMem-v1`, revision `a8076d5608c93ba2a28983cd78aa99b01a163ae7`. **Licence:** MIT. v2 and v3 now exist; we use v1, the version in the paper (arXiv:2504.14225).
- **Files:**
  - `questions_{32k,128k,1M}.csv` has columns `persona_id, question_id, question_type, topic, context_length_in_tokens, ..., user_question_or_message, correct_answer ("(c)"), all_options (JSON list of 4 "(a) ..." strings), shared_context_id, end_index_in_shared_context`. The 32k split has 589 questions in 7 question types.
  - `shared_contexts_{split}.jsonl` has one line per context: `{context_id: [{role, content}, ...]}`. The first message is a system message holding the persona.
- **Context for a question:** `contexts[shared_context_id][:end_index+1]`. The metric is multiple-choice accuracy.

## LoCoMo (E8)
- **Source:** github.com/snap-research/locomo `data/locomo10.json` (main branch; sha256 stored in raw_manifest). **Licence:** GitHub does not detect a licence (NOASSERTION), so check the repository README before redistributing anything.
- **Structure:** 10 conversations. Each has `{sample_id, conversation{speaker_a, speaker_b, session_<i>[{speaker, dia_id, text, ...}], session_<i>_date_time}, qa[{question, answer | adversarial_answer, evidence, category}], event_summary, observation, session_summary}`.
- **QA:** categories 1–5, with 1,986 QA in total. Category 5 is adversarial and is excluded (D-027), leaving 1,540 QA.
