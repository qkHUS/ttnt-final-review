# Dataset Setup

> **Tutorial:** a full walkthrough of the data-preparation + TF-IDF skill-extraction pipeline
> (for readers with zero background, in Vietnamese) lives in
> [`docs/bai-giang-trich-xuat-skill-tfidf.md`](../docs/bai-giang-trich-xuat-skill-tfidf.md)
> — PDF: `docs/bai-giang-trich-xuat-skill-tfidf.pdf` (20 pages).
> Regenerate it with `python docs/md2pdf.py`.
> **Report (data + skill extraction + pair scoring):**
> [`report/bao-cao-workflow-tf-idf.md`](../report/bao-cao-workflow-tf-idf.md)
> — PDF: `report/bao-cao-workflow-tf-idf.pdf` (14 pages).

## 1. Dataset

This project uses two datasets:

### Parser Dataset

* Dataset: `yashpwr/resume-ner-training-data`
* Purpose: Used to train and evaluate the CV Parser / NER model.
* Source: Hugging Face Datasets

### Matching Dataset

* Dataset: `med2425/resume-job-fit-merged-v1`
* Purpose: Used to train and evaluate the Resume–Job Description Matching model.
* Source: Hugging Face Datasets

The datasets are downloaded automatically from Hugging Face and are stored in the local Hugging Face cache. They are not committed to GitHub.

## 2. Environment Setup

Make sure Python 3.10+ is installed.

Check Python:

```bash
python --version
```

## 3. Download Dataset

From the project root, run:

```bash
python data/download_data.py
```

The script automatically downloads both datasets from Hugging Face.

After downloading, the datasets can be loaded in the project using:

```python
from datasets import load_dataset

parser_data = load_dataset("yashpwr/resume-ner-training-data")

matching_data = load_dataset("med2425/resume-job-fit-merged-v1")
```

The downloaded datasets are cached locally by Hugging Face and do not need to be downloaded again unless the cache is removed.

## 4. Derived files (data preparation)

> **Scope note (pipeline decision).** The extraction & matching pipeline uses **only** the
> matching source: `cv_docs.csv` (643 CV) + `jd_docs.csv` (351 JD). The parser source
> (`parser_train.csv`, 22,855 chat rows where each CV is embedded inside prompts) is
> **not part of this pipeline** — its derived tables (`parser_docs.csv`,
> `parser_contact_pairs.csv`, `build_parser_docs.py`) have been **removed from `data/`**
> (provenance: 15,092 unique CVs, 0 overlap with the 643 matching CVs, 172 contact
> pairs — rebuildable from the raw CSV if ever needed). The raw `parser_train.csv`
> download is kept solely as the data source for the separate CV Parser / NER milestone.

All CSV files in `data/` are **not committed to git** (see `.gitignore`); rebuild them with the scripts below, run from the project root.

### Raw exports (from `download_data.py`)

| File | Rows | Content |
|---|---|---|
| `parser_train.csv` | 22,855 | Chat-format multi-task resume data (summarize / categorize / rewrite / critique / contact-extract...). CV text lives inside user messages. **No token-level NER labels** despite the dataset name. |
| `matching_train.csv` | 80,017 | Pairs: `resume`, `jd`, `label` (Good/Potential/No Fit), `source`, `resume_domain`, `jd_domain`. Only **642 unique CVs** and **280 unique JDs** — pairs are combinations. |
| `matching_test.csv` | 13,716 | Same schema. 476/477 of its CVs also appear in train (official split leaks CVs across sides). |

### Document & pair tables (from `build_pairs.py`)

Run: `python data/build_pairs.py`

| File | Rows | Content |
|---|---|---|
| `pairs.csv` | 93,733 | One row per labeled pair: `pair_id`, `resume_id`, `jd_id`, `label`, `split`, domains. No text — join to the doc tables by id. |
| `cv_docs.csv` | 643 | Unique CVs: `doc_id` (md5 of raw text), `n_train_pairs`, `n_test_pairs`, `text` (PII masked). |
| `jd_docs.csv` | 351 | Unique JDs, same schema. |

`doc_id` = first 12 hex chars of md5(text): stable across machines and reruns, so extraction outputs (`data/extracted/<set>/<doc_id>.json`) can be produced **once per document** and joined back onto `pairs.csv` without recomputing anything.

### Skill lexicon

`skills_lexicon.txt` is seeded automatically by `extract_json.py` (delete it to rebuild); `skills_lexicon_extra.txt` comes from `python data/build_lexicon_extra.py` (needs network once; raw files cached in `data/_hf_tmp/`).

| File | Rows | Content |
|---|---|---|
| `skills_lexicon.txt` | 1,176 | Base skill vocabulary seeded from the trailing "Skills" blocks of the 643 matching CVs (items appearing in ≥3 CVs; 10 generic words like "state", "client" are blocked in code via `STOP_SKILLS` when loaded). Seeding also normalizes glued PDF text: a `·` bullet becomes a space and `category:skill` fragments keep only the part after the colon ("big data technologies:hadoop" → "hadoop"). Hand-editable; `extract_json.py` reuses this file if present, so manual additions survive reruns (delete the file to rebuild from `extract_json.py`'s `build_lexicon`). |
| `skills_lexicon_extra.txt` | 6,108 | Additional skills from sources **outside** the matching corpus, so no leakage: `batuhanmtl/job-skill-set` (4,559 skills parsed from 1,167 JDs' `job_skill_set` lists), `tabiya/esco_skills_test` skillspan (1,445 spans/sub-spans labelled with ESCO), techwolf (267 ESCO labels). Filtered to the same shape rules as the pseudo-GT (2–40 chars, ≤4 words, has letters, not in `STOP_SKILLS`/junk), trailing punctuation stripped, ". NET" → ".net" collapsed, placeholder `<...>` entries dropped. |

`extract_json.py` automatically loads this file on top of `skills_lexicon.txt` (both in extraction and in the held-out eval — external skills never saw any matching CV text). Combined lexicon: **7,137 skills** (1,146 of the 1,176 self-seeded rows survive the code-level `STOP_SKILLS` block, which also filters US state abbreviations and company suffixes like `inc`/`llc` that leak into Skills tails from glued addresses).

### Skill extraction — structured JSON (from `extract_json.py`)

Run: `python data/extract_json.py`

Supersedes the former `extract_tfidf.py` / `eval_external.py` / `skills_tfidf.csv` pipeline (deleted 2026; held-out protocol [A] below still reproduces its methodology, its exact numbers now differ because the lexicon and pseudo-GT were improved since). One JSON file per document, `data/extracted/<set>/<doc_id>.json`:

| Dir | Files | Schema |
|---|---|---|
| `extracted/cv/` | 643 | `{doc_id, name, skills[], experience_years, education{degree, major, university}}` |
| `extracted/jd/` | 351 | `{doc_id, job_title, required_skills[], preferred_skills[], required_years}` |
| `extracted/test_cv/` | 999 | CV schema applied to the external CV stream (1 broken-text CV dropped) |
| `extracted/test_jd/` | 100 | JD schema applied to the external JD stream |

`skills` / `required_skills` reuse the TF-IDF machinery of the previous pipeline unchanged: top-25 lexicon-ranked TF-IDF terms (n-gram 1-2, min_df=3, sublinear_tf) plus a presence scan of all 7,137 lexicon skills with a lookaround regex (so `c++`/`.net`/`c#` are caught despite `\b` tokenizers; nested duplicates like "sql" inside "sql server" are deduplicated). Scan hits found inside the "Skills" region (1,500 chars after the last Skills header, or on a bullet line) are ordered before hits found elsewhere in the document, so the head of the list is precision-focused. JD skills are split required/preferred by the position of the "nice to have / preferred / bonus" marker. The remaining fields are regex parsers: `name` (strict first-line person-name rule — **0% on the matching corpus**: its generated CVs are glued single-line texts that carry no name line at all, so `null` is the honest value; the same rule finds names in 10.1% of the real external PDF resumes, where resumes genuinely start with a name), `experience_years` ("N years … experience" + overlapping date-range fallback, 22.2%), `education` degree/major/university (99.7% / 62.1% / 93.0%), JD `job_title` ("Job Title:" line, 51.0%) and `required_years` (66.7%).

Held-out evaluation — same protocol as before (80/20 CV split seed 42; lexicon + IDF fitted on the 80% only; lexicon = self-seeded + external; 105 of 129 held-out CVs carry a pseudo-GT "Skills" block). Two protocols are reported because the JSON output contains more than the top-25 list:
- **[A] legacy protocol** — top-25 TF-IDF∩lexicon list, exact string match: **P@25 = 43.4%, R@25 = 28.1%** (this is what the earlier 39.8/25.3 numbers measured, before the lexicon/GT improvements).
- **[B] real output** — the full merged `skills` list scored with containment matching (a GT phrase counts as found when it appears inside an extracted skill at a word boundary), top-25 window and full list: **P@25 = 51.0%, R@25 = 44.7%, R@full = 74.6%**.

Known limits (unchanged): no semantics (no "sql" ≈ "sql server"), and the pseudo-GT is circular — the lexicon (and its residual junk) is built from the same "Skills" blocks used as ground truth, so this metric underestimates real-world precision but is the honest, reproducible baseline the BERT route must beat.

Improvement experiments on the same held-out set (all measured with protocol [B] against the topk-only baseline 43.4/28.1; adopted items in bold): real output vs topk-only +1.5 R@25; **containment matching +11.6 R@25 and R@full 28.1→74.6** (the JSON list is much longer than 25, so recall must be scored on the whole list); **tier ordering (Skills-region hits first) +3.8 P@25 +5.1 R@25**. Lexicon/GT hygiene (min_docs 5→3, glued-text parse fix `category:skill`→`skill`, state-abbreviation/company-suffix blocking) is what lifted the legacy baseline itself from 39.8/25.3 to 43.4/28.1. Rejected: plural-variant scanning (no gain), generic-skill df filter (R@full drops to 64–67% — generic skills do appear in GTs), min_docs=2 (+542 entries, slightly lower P/R@25), adding test-CV "Skills" blocks to the lexicon (no gain, keeps the external stream clean). Earlier rounds of the deleted pipeline had already rejected alias expansion, multi-word score boost and a minimum-score threshold.

### External test set (from `build_testset.py`, extracted by `extract_json.py`)

Run: `python data/build_testset.py` then `python data/extract_json.py` (network once; raw cache in `data/_hf_tmp/`)

A hold-out stream **from sources never used anywhere above**, to test whether the extraction pipeline works on data it was not built from:

| File | Rows | Content |
|---|---|---|
| `test_cv_docs.csv` | 1,000 | CVs from `d4rk3r/resumes-raw-pdf` (real resumes as PDFs, text extracted with pypdf, ≥500 chars, deduped; 929 CVs) topped up with the Kaggle `Sachinkelenjaguri/Resume_dataset` (962 rows but only 166 unique — 71 unique CVs used): `doc_id`, `source`, `category`, `text` (PII masked). |
| `test_jd_docs.csv` | 100 | JDs sampled (seed 42) from `jacob-hugging-face/job-descriptions` (853 rows, ≥200 chars): `doc_id`, `source`, `position_title`, `company_name`, `text`. |
| `extracted/test_cv/` | 999 | CV-schema JSON per document (1 broken-text CV dropped). |
| `extracted/test_jd/` | 100 | JD-schema JSON per document. |

Overlap with the main corpus: **1/1000 CV, 0/100 JD** share a `doc_id` — a genuinely independent stream.

Coverage of the pipeline applied to unseen data (lexicon + TF-IDF fitted on the 994 main-corpus docs only, then applied as-is — deployment-realistic; skill count = merged `skills` list for CVs, `required`+`preferred` for JDs):

| Set | ≥1 skill | ≥5 | ≥10 | avg skills/doc | zero-skill docs |
|---|---|---|---|---|---|
| 999 external CVs | 94.3% | 73.2% | 57.8% | 26.5 | 57 |
| 100 external JDs | 100% | 99.0% | 94.0% | 37.3 | 0 |

All 57 zero-skill CVs are short, broken PDF text extractions (avg 1,237 chars vs 3,820 for the rest; split Vietnamese diacritics, private-use glyphs) — text-extraction artifacts of the PDF source, not pipeline misses. 11.5% of the external CVs carry a parseable "Skills" block, reserved as pseudo-GT for a future external precision check.

### Pair scoring — fit / no-fit (from `match_score.py`)

Run: `python -X utf8 data/match_score.py` (reuses the exact TF-IDF fitted by `extract_json.py` on the 994 main-corpus docs — nothing is re-fitted on pairs).

Features per pair (93,733): `cosine` (dot product of the L2-normalized TF-IDF vectors), `req_cov` / `pref_cov` — share of the JD's required/preferred skills present in the CV, matched with boundary-safe 1–4-gram sets (NaN when the JD has no list, 0 is *not* used).

Outputs:

| File | Rows | Content |
|---|---|---|
| `scored/pairs_scored.csv` | 93,733 | `pair_id, resume_id, jd_id, label, split, cosine, req_cov, pref_cov` — ready for downstream ranking |
| `scored/demo_matches.json` | 3 | one Good / Potential / No-Fit pair of the same JD with matched & missing skill lists |

Measured on the 13,716 test pairs, thresholds frozen on train (Good Fit prevalence 12.2%):

- Label means — cosine: 0.0654 (Good) / 0.0503 (Potential) / 0.0326 (No Fit); req_cov: 0.2615 / 0.2198 / 0.1425.
- Good vs rest: cosine alone **AUC 0.805, PR-AUC 0.433** (t*=0.064 → F1 0.443); logistic regression on all 3 features **AUC 0.810, PR-AUC 0.443** (3.6× prevalence), F1 0.455.
- 3-band split (t_lo=0.0459, t_hi=0.0639, chosen on train): accuracy 0.634, macro-F1 0.516 — vs majority baseline 0.617.

Full teaching write-up with formulas, per-function code walkthrough, and references: [`report/bao-cao-workflow-tf-idf.md`](../report/bao-cao-workflow-tf-idf.md).
