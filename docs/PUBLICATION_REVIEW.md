# Publication review pending

No GitHub creation, push, archive, deletion, visibility change or history rewrite was performed. Local preparation does not establish publication rights.

The historical-object audit covered 17 cleaning blobs and 82 matching blobs across all 11 original commits. Token/private-key/assigned-secret patterns found no matching credential. File inventory did not identify a tracked case database, Excel workbook, confidential research report or API-key file. This limited audit cannot establish confidentiality or copyright status from source text alone.

Two legacy sources need provenance confirmation before the inherited history can be republished:

1. `db_cleaning`, original `tests/test_pipeline.py`, full-name fixture in the human-correction test. Its origin is not documented. The current test uses a newly fabricated replacement, but the old value remains in inherited commits.
2. `id_matching`, original `identity_matching_tool/data/sample_clean_missing_persons.csv`, and related record fixtures. The records look like examples, but the repository does not explicitly document that every row is fabricated or authorized. The current sample has been replaced; original history is still reachable.

The pre-existing local `CICR_Missing_ppl/Base-Consolidada-pfr.xlsx` was not opened, copied, added to Git or modified. It remains outside these two repositories.

The owner must confirm that historical record fixtures are synthetic or authorized for publication, and that publication of the PIR code and ICRC contextual attribution is permitted. No license has been added because these rights cannot be inferred from public accessibility. If confirmation is unavailable, a separate sanitized publication history can be prepared while retaining the original clones/bundles; that choice needs explicit authorization. No existing history has been silently rewritten.

Operational caches include case IDs, names/IPA or blocking signatures when real records are processed. Keep those files private, including caches stored via MATCHING_CACHE_DIR outside the default ignored directory. Git ignores are preventive, not a data anonymization mechanism.
