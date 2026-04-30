# Identity Matching Tool

Application Streamlit locale pour analyser des correspondances potentielles entre dossiers de personnes disparues.

## Fonctionnement

L'application demande un CSV propre, verifie son schema, affiche un resume qualite, puis transforme les lignes valides en dossiers internes. L'utilisateur choisit ensuite un mode d'analyse :

- `Single case matching` : recherche les dossiers les plus proches d'un dossier cible ;
- `Global matching` : compare les dossiers entre eux pour faire ressortir les meilleures correspondances.

Le moteur peut combiner plusieurs familles de scores : nom, phonetique, localisation et temporalite. Le blocking limite le nombre de paires a scorer, puis les resultats sont tries par score. Les caches locaux accelerent les relances, notamment pour les scores de paires, les index de blocking et les conversions phonetiques IPA.

Ces scores sont des aides a l'analyse, pas des preuves.

## Format du CSV

Colonnes obligatoires :

- `case_id`
- `full_name_raw`
- `full_name_normalized`
- `name_tokens`
- `disappearance_city_verified`
- `latitude`
- `longitude`
- `disappearance_date_parsed`

Colonnes recommandees :

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

Une colonne obligatoire manquante bloque l'analyse. Une colonne recommandee manquante affiche seulement un avertissement.

## Installation sans venv

Depuis ce dossier :

```powershell
python -m pip install --target python_packages -r requirements.txt
```

Les paquets sont installes dans `python_packages/`. Ce dossier n'est pas versionne et peut etre regenere sur une autre machine.

## Lancement

Depuis ce dossier :

```powershell
$env:PYTHONPATH = "$PWD\python_packages"
python -m streamlit run app.py
```

Un fichier d'exemple est disponible dans `data/sample_clean_missing_persons.csv`.

## Tests

```powershell
$env:PYTHONPATH = "$PWD\python_packages"
python -m pytest
```

## Fichiers generes

L'application peut creer des fichiers dans `cache/` :

- `pairwise_scores.csv` : cache des scores de paires ;
- `phonetic_ipa_cache.csv` : cache des conversions phonetiques ;
- `blocking_indexes/` : exports de diagnostic des buckets de blocking.

Ces fichiers, les `__pycache__`, `.pytest_cache` et `python_packages/` sont ignores par Git car ils sont regenerables.
