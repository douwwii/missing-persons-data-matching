# Fabricated demonstration data

All rows in `synthetic_persons.csv`, `synthetic_cities.csv` and `synthetic_clean.csv` were newly invented for this demonstration. Names, case IDs, places, dates and coordinates do not describe an actual investigation. Any resemblance to an individual or location is coincidental.

For the cleaning interface, upload `synthetic_persons.csv` and map `id → case_id`, `name → full_name`, `city → disappearance_city`, `department → disappearance_department`, and `date → disappearance_date`. Upload `synthetic_cities.csv` as the reference table and map its four corresponding location fields. Exact locations resolve automatically; the intentionally misspelled `Testvile` requires review. Select `Testville` for the fourth record, or leave it unresolved to explore missing geography.

Export the clean CSV and upload it to the matching interface. Alternatively, upload the included `synthetic_clean.csv` directly. Start with exhaustive global matching, enable name/location/time criteria, and inspect the suggested pairs and their components. Turn on phonetics only with its optional dependencies installed; the demo uses `spa-Latn` to exercise the implemented default.

`python -m examples.demo` performs the same file exchange programmatically and records an explicit human decision for the fourth row. It creates ignored output under `generated/` and local cache files under `cache/`. `python -m examples.demo --phonetic` exercises the real phonetic path as well.

The invented near-match between records `0001` and `0002` is an illustration, not ground truth or a scientific result.
