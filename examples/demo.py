"""Run the preparation-to-matching workflow using only fabricated fixtures."""

import argparse
from pathlib import Path

from data_cleaning.cities import build_city_references
from data_cleaning.io import read_table
from data_cleaning.mapping import ColumnMapping
from data_cleaning.pipeline import HumanDecision, build_clean_database
from identity_matching.data_loading.clean_file_loader import dataframe_to_records, load_clean_csv
from identity_matching.data_loading.schema_validator import validate_clean_dataset
from identity_matching.matching.matching_engine import run_matching
from identity_matching.models import AnalysisConfig, ParallelConfig

EXAMPLES = Path(__file__).resolve().parent


def clean_synthetic_records():
    """Normalize fabricated records and demonstrate one explicit human decision."""
    persons = read_table(EXAMPLES / "synthetic_persons.csv")
    cities = read_table(EXAMPLES / "synthetic_cities.csv")
    references = build_city_references(cities, "city", "department", "latitude", "longitude")
    mapping = ColumnMapping(
        {
            "id": "case_id",
            "name": "full_name",
            "city": "disappearance_city",
            "department": "disappearance_department",
            "date": "disappearance_date",
        }
    )
    decisions = {"0004": HumanDecision("accept_candidate", reference_id=references[0].ref_id)}
    return build_clean_database(persons, mapping, references, decisions)


def run_demo(output_dir: Path, phonetic: bool = False):
    """Export a clean CSV, reload it through the real loader, and rank all pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_clean.csv"
    clean_synthetic_records().to_csv(path, index=False)
    dataset = load_clean_csv(path)
    validation = validate_clean_dataset(dataset)
    if not validation.is_valid:
        raise ValueError(validation.message)
    records = dataframe_to_records(dataset)
    config = AnalysisConfig(
        investigation_mode="global",
        analysis_mode="exhaustive",
        enabled_filters={"name": True, "phonetic": phonetic, "geographic": True, "temporal": True},
        filter_parameters={"min_score_threshold": 0.7, "max_results": 10, "phonetic_lang": "spa-Latn"},
    )
    result = run_matching(records, config, parallel_config=ParallelConfig(num_workers=1))
    print(f"Clean file: {path}")
    print(f"Fabricated records: {len(records)}; pairs compared: {result.compared_pairs_count}")
    for pair in result.results:
        print(f"{pair.id_A} / {pair.id_B}: {pair.score_final:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLES.parent / "generated")
    parser.add_argument("--phonetic", action="store_true", help="Also use Epitran/PanPhon with spa-Latn.")
    args = parser.parse_args()
    run_demo(args.output_dir, args.phonetic)


if __name__ == "__main__":
    main()
