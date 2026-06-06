# Réadmissions hospitalières — Prédiction multiclasse & aide à la décision

Modélisation du risque de réadmission à 30 jours sur données diabétiques (UCI).  
Cible : `readmitted ∈ { "<30", ">30", "NO" }`

Périmètre couvert : audit des données, segmentation, modélisation multiclasse, interprétabilité SHAP, dashboard opérationnel et plan d'expérimentation A/B.

---

## Structure du projet

```
.
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv          # séjours hospitaliers (~100 k lignes)
│   │   └── IDS_mapping.csv            # mapping identifiants / métadonnées
│   └── processed/
│       ├── dataset_model.csv          # dataset nettoyé, prêt pour la modélisation
│       ├── dataset_model_with_clusters.csv   # idem + colonne cluster
│       └── diag_category.csv          # mapping diagnostic → catégorie clinique
├── notebooks/
|   ├── 00_process_data.ipynb
│   ├── 01_business_case_readmissions.ipynb   # définition du problème, KPI tree, ROI
│   ├── 02_data_audit.ipynb        # audit qualité, dictionnaire, nettoyage
│   ├── 03_eda_and_tests.ipynb          # KMeans, Elbow/Silhouette, profils segments
│   ├── 04_models.ipynb       # modèles sklearn, validation croisée, métriques
│   └── 05_shap.ipynb         # SHAP global/local, recommandations opérationnelles
├── dashboard/
│   └── app.py                         # Streamlit — 5 vues (KPIs, EDA, Segments, Modèle, SHAP)
├── reports/
│   └── figures/                       # graphiques exportés par les notebooks
├── abtest_plan/
│   └── abtest_plan.md                 # protocole expérimental A/B test
├── requirements.txt
└── README.md
```

---

## Données

### Fichiers source

| Fichier                            | Description                                                   |
| ---------------------------------- | ------------------------------------------------------------- |
| `data/raw/diabetic_data.csv`       | Séjours hospitaliers — features cliniques et administratives  |
| `data/raw/IDs_mapping.csv`         | Tables de correspondance (codes → libellés)                   |
| `data/processed/diag_category.csv` | Mapping diagnostic → catégorie clinique (créé en notebook 02) |

Source originale : [UCI ML Repository — Diabetes 130-US hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

### Fichiers générés

`dataset_model.csv` et `dataset_model_with_clusters.csv` sont produits par les notebooks et ne sont pas versionnés si leur taille dépasse les limites du dépôt. Dans ce cas, seul `diag_category.csv` est versionné ; les autres se régénèrent en exécutant les notebooks dans l'ordre.

---

## Installation

### 1. Créer et activer un environnement virtuel

```bash
python -m venv .venv
```

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Exécution des notebooks

```bash
jupyter lab
```

## Dashboard Streamlit

Le dashboard lit `data/processed/dataset_model.csv` par défaut (chemin modifiable dans la barre latérale).

```bash
streamlit run dashboard/app.py
```

## Plan A/B test

Le protocole est décrit dans `abtest_plan/abtest_plan.md` : définition des groupes, métriques primaires et secondaires, calcul de la taille d'échantillon et critères d'arrêt.

---

## Dépendances principales

| Librairie           | Usage                                   |
| ------------------- | --------------------------------------- |
| pandas, numpy       | Manipulation des données                |
| scikit-learn        | Modélisation, clustering, prétraitement |
| shap                | Interprétabilité du modèle              |
| streamlit, plotly   | Dashboard interactif                    |
| matplotlib, seaborn | Visualisations dans les notebooks       |
| jupyter             | Exécution des notebooks                 |

Liste complète dans `requirements.txt`.
