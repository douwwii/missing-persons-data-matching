# Nettoyage de base pour identity matching

Cette application prépare une base brute de personnes disparues pour une future étape d'identity matching.

Elle permet de :
- importer une base de dossiers et une base de référence de villes ;
- mapper manuellement les colonnes ;
- normaliser noms, villes, départements et dates ;
- associer les villes à des coordonnées issues uniquement de la base de référence ;
- traiter les conflits avec validation humaine ;
- exporter un CSV final stable pour l'étape 2.

## Installation

```powershell
python -m pip install -r requirements.txt --target .\python_packages
```

## Lancer l'interface

```powershell
$env:PYTHONPATH="$PWD\python_packages"
python -m streamlit run app.py
```

Ouvrir ensuite :

```text
http://127.0.0.1:8501
```

Si `localhost` charge en boucle, utiliser `127.0.0.1`, qui evite certains blocages de resolution locale ou de WebSocket sous Windows.

Si le port 8501 est deja utilise :

```powershell
$env:PYTHONPATH="$PWD\python_packages"
python -m streamlit run app.py --server.port 8502
```

## Lancer les tests

```powershell
$env:PYTHONPATH="$PWD\python_packages"
python -m pytest
```
