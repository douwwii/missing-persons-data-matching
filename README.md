# Identity Matching

Application locale Streamlit pour aider a rechercher des correspondances potentielles entre dossiers de personnes disparues.

Le code de l'application est dans `identity_matching_tool/`.

## Fonctionnement

L'application charge un fichier CSV propre, valide les colonnes attendues, transforme les lignes en dossiers internes, puis permet de lancer une recherche de correspondances en mode dossier cible ou en mode global.

Les scores combinent selon la configuration choisie :

- similarite des noms ;
- similarite phonetique ;
- distance geographique ;
- proximite temporelle.

Les resultats sont des aides a l'analyse. Ils ne remplacent pas la decision d'un enqueteur.

## Lancer l'application

Depuis la racine du projet :

```powershell
cd identity_matching_tool
python -m pip install --target python_packages -r requirements.txt
$env:PYTHONPATH = "$PWD\python_packages"
python -m streamlit run app.py
```

Cette installation place les paquets dans `identity_matching_tool/python_packages/` au lieu d'utiliser une venv. Ce dossier est ignore par Git et peut etre regenere sur une autre machine avec la meme commande.

Un CSV d'exemple est disponible dans `identity_matching_tool/data/sample_clean_missing_persons.csv`.

## Tests

```powershell
cd identity_matching_tool
$env:PYTHONPATH = "$PWD\python_packages"
python -m pytest
```
