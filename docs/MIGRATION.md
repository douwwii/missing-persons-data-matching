# Audit and migration

Original sources: `douwwii/db_cleaning` at `ec26129` (2 commits) and `douwwii/id_matching` at `4a94e19` (9 commits). Both histories are ancestors of the unified branch, linked without changing original commits. Original clones and bundles remain under the portfolio's `.migration/` directory. No original repository has been deleted or archived.

Migration plan implemented: retain the cleaning history as the base, import the full matching tree under a temporary prefix with an unrelated-history merge, then move the applications into distinct `data_cleaning` and `identity_matching` packages. Historical documentation is retained under `docs/history` with a notice that its paths and commands are obsolete.

The original matching application could not import `src.cache.sqlite_cache`: the connection module was absent from all reachable history. Its broad `cache/` ignore rule would also ignore Python source inside such a directory. The restored module handles row-based SQLite results, timeout/WAL configuration, transactions and connection closure; `/cache/` now ignores runtime data without hiding source code.

Other corrections: cleaner token strings now parse as individual matching tokens; readers retain zero-padded identifiers; missing/duplicate IDs block the matching UI; invalid/nonfinite reference coordinates are excluded; XLS support explicitly declares xlrd; multiprocessing uses spawn to avoid forking Streamlit threads. Core and optional phonetic dependencies are centralized in pyproject.toml with resolved requirement files. Existing algorithms, two interfaces, refinement and caching behavior are retained and tested.

The legacy sample CSV was removed from the current tree and replaced with newly fabricated fixtures; one test's full-name fixture of undocumented provenance was replaced. Originals and historical commits remain intact. This is not historical data erasure: see `PUBLICATION_REVIEW.md` before any push.

Before correction, 18 cleaning tests passed and the matching suite failed collection with 7 missing-cache import errors. After migration, all 127 original tests passed; new integration and application startup tests extend coverage. Results and remaining verification limits are recorded in `VERIFICATION.md`.
