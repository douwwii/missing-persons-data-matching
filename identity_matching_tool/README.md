# Identity Matching Tool

Local Streamlit application for analyzing potential matches between missing person case records.

## Workflow

The application loads a clean CSV, validates its schema, displays a quality summary, then converts valid rows into internal case records. The user can then choose an analysis mode:

- `Single case matching`: finds the closest case records for one target case;
- `Global matching`: compares case records against each other to surface the strongest potential matches.

The engine can combine several score families: name, phonetic, location, and time. Blocking limits the number of pairs to score, then results are sorted by score. Local SQLite caches speed up reruns, especially for pair scores, blocking buckets, and IPA phonetic conversions.

Scores are investigative aids, not evidence.

## Expected CSV Format

Required columns:

- `case_id`
- `full_name_raw`
- `full_name_normalized`
- `name_tokens`
- `disappearance_city_verified`
- `latitude`
- `longitude`
- `disappearance_date_parsed`

Recommended columns:

- `first_name_raw`
- `first_name_normalized`
- `last_name_raw`
- `last_name_normalized`
- `disappearance_city_raw`
- `disappearance_city_normalized`
- `disappearance_department_verified`
- `disappearance_date_raw`
- `city_resolution_status`
- `city_resolution_score`
- `city_resolution_explanation`

A missing required column blocks the analysis. A missing recommended column only displays a warning.

## Install Without A Virtualenv

From this directory:

```powershell
python -m pip install --target python_packages -r requirements.txt
```

Packages are installed in `python_packages/`. This directory is not versioned and can be regenerated on another machine.

## Run

From this directory:

```powershell
$env:PYTHONPATH = "$PWD\python_packages"
python -m streamlit run app.py
```

A sample file is available at `data/sample_clean_missing_persons.csv`.

## Tests

```powershell
$env:PYTHONPATH = "$PWD\python_packages"
python -m pytest
```

## Cache Strategy

The application uses `cache/identity_matching_cache.sqlite` as the operational cache.

It contains three cache families:

- IPA cache: one reordered IPA value per case record, language, normalized name hash, and phonetic algorithm version.
- Blocking bucket cache: for a dataset and blocking configuration, stores only buckets such as `R1`, `R2`, and `R3` bucket keys mapped to case IDs. Candidate pairs are regenerated from buckets at analysis time.
- Selective pairwise component cache: stores reusable pair components only when at least two criteria have a score strictly above `0.5`. Stored components include name scores, phonetic scores, raw GPS distance, and raw date gap. Final scores are rebuilt from these components with the current filter parameters, so changing a geographic or temporal threshold does not force stable components to be recomputed.

Older final-score pair caches are no longer used. The selective pairwise component cache is the source of truth for pair reuse; `strong_criteria_count` records how many dimensions passed the `0.5` cache threshold.

When only one scoring filter is enabled, the engine also computes lightweight location and time components, when data is available, only to decide whether the pair satisfies the selective cache policy. These supplemental scores do not change the final score unless their filters are enabled.

Diagnostic blocking bucket exports may also be written under `cache/blocking_indexes/` so large buckets can be inspected manually.

Generated cache files, `__pycache__`, `.pytest_cache`, and `python_packages/` are ignored by Git because they are regenerable.
