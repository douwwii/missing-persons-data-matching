# Verification record

Verification performed on Fedora on 2026-10-13:

| Check | Result |
| --- | --- |
| Original cleaning baseline | 18 passed |
| Original matching baseline | Collection failed: 7 errors caused by the absent SQLite connection module |
| Unified suite, Python 3.12.15 with phonetic dependencies | 140 passed |
| Unified core suite, Python 3.14.8 without phonetic dependencies | 139 passed, 1 real-library phonetic test skipped |
| Ruff 0.17.0 lint/format | Passed |
| Package installation and dependency compatibility | Editable installation succeeded; uv reported all 56 installed packages compatible |
| Cleaning and matching Streamlit startup | Both AppTest sessions had no exceptions |
| Synthetic CSV round trip | Four text IDs preserved; tokens parsed individually; six pairs compared |
| Real Epitran/PanPhon path | Identical invented names scored 1; a different invented name scored lower; spa-Latn conversion and distance completed |

The inherited phonetic mocks reused a global distance object across tests, masking an outdated `> 0.80` expectation. Each test now resets this object; the test's one-edit example is checked against the documented power-calibrated score (~0.7684). The algorithm itself was not changed to make the test pass. The real-library test separately checks resource loading and a nonidentical pair.

Python 3.14 phonetic installation failed while building `editdistance`, whose compilation needed unavailable Python development headers. Python 3.12 used an available binary wheel and completed installation. A private standalone Python 3.12 can be used without modifying Fedora's system Python.

This verifies startup and core processing, not every browser interaction or the accuracy of real case matching. No operational ICRC dataset, confidential document or pre-existing local workbook was used. No evaluation metric on real people has been inferred from these checks.

Final publication review added a CSV round-trip regression test for quality summaries: NaN fields are counted as missing after reload. The full suites passed with Python 3.12 and 3.14; the latter skips the optional real-library phonetic test.

Case-ID import correction: 153 tests passed under Python 3.12, and the 13 new case-ID tests passed under Python 3.14. Synthetic checks cover duplicate/missing IDs, preservation of 10,290 rows, collisions with existing IDs, source-reference backups, all-pairs matching after repair, literal NA/NULL identifiers, the explicit Streamlit repair button, schema failures and subsequent uploads. No real case database was opened or altered.
