# RationalML — fondations V0.1.1

API Python pour la classification binaire, avec optimisation Optuna sur le train
uniquement. Les six briques publiques sont `TaskType`, `AutoMLConfig`,
`MetricRegistry`, `ModelRegistry`, `AutoML` et `AutoMLResult`.

## Installation et utilisation

Depuis ce dossier, avec Python 3.10 ou plus :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[boosting,legacy,test]"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Le cœur requiert numpy, pandas, scikit-learn, Optuna et threadpoolctl. LightGBM et
XGBoost sont dans l'extra `boosting`, chargé à la demande. `models="auto"`
sélectionne les modèles enregistrés dont les dépendances sont installées ; un
modèle explicitement demandé mais absent produit `MissingDependencyError`.
L'extra `legacy` fournit tqdm pour `local_optimizer`.

```python
from rationalml import AutoML, AutoMLConfig

result = AutoML(
    target="churn",
    task="auto",
    metric="roc_auc",
    models=["logistic_regression", "lightgbm", "xgboost"],
    cv=5,
    n_trials=30,
    random_state=42,
    positive_class=1,  # pour une cible 0/1
).fit(df)

predictions = result.predict(df.drop(columns="churn"))
probabilities = result.predict_proba(df.drop(columns="churn"))
positive_probabilities = result.predict_positive_proba(df.drop(columns="churn"))
leaderboard = result.leaderboard
test_metrics = result.test_metrics

# Variante avec configuration indépendante des données :
config = AutoMLConfig(target="churn", models="logistic_regression")
result = AutoML(config).fit(df)
```

`fit` exige un DataFrame avec des colonnes uniques, des features numériques ou
booléennes finies et une cible sans valeur manquante à exactement deux classes.
Aucune ligne n'est retirée, aucune feature n'est imputée, transformée ou
discrétisée. La validation signale les données à préparer explicitement.
Chaque classe doit conserver au moins `cv` observations dans le train, et les
deux classes doivent être présentes dans le test.

À l'inférence, les colonnes du DataFrame doivent avoir les mêmes noms et le même
ordre ; une matrice numpy de même largeur est aussi acceptée.

## Contrat de classe positive binaire

`AutoMLConfig.positive_class: object | None = None` est aussi accepté directement
par `AutoML`. L'appartenance aux deux valeurs de la cible est vérifiée avant
Optuna, à partir du train. L'encodage travaille sur une copie des labels :

- Avec une valeur explicite, cette classe est toujours encodée en **1** et
  l'autre en **0**, quel que soit leur ordre alphabétique ou numérique. Une
  valeur absente de la cible produit `DataValidationError`. Les valeurs `0`,
  `False` et les chaînes vides sont des choix explicites valides si présents.
- Sans valeur explicite, une cible booléenne utilise `True` comme positive,
  et une cible numérique exactement `{0, 1}` utilise `1`, sans avertissement.
- Pour les autres labels comparables, le plus grand label selon leur ordre
  naturel est choisi de manière déterministe, en conservant le comportement
  V0.1. Un `UserWarning` précise la classe choisie et recommande de renseigner
  `positive_class`. Ce choix ne dépend ni des fréquences ni de l'ordre des lignes.

L'encodage binaire n'utilise plus `LabelEncoder`. `AutoMLResult` expose :

- `positive_class` : label original correspondant à la classe interne 1 ;
- `negative_class` : label original correspondant à la classe interne 0 ;
- `classes_` : les labels originaux dans l'ordre **[negative_class, positive_class]**.

`result.predict(X)` restitue les labels originaux. `result.predict_proba(X)`
retourne une matrice de forme `(n_lignes, 2)`, avec **P(negative_class) en colonne
0** et **P(positive_class) en colonne 1**. `predict_positive_proba(X)` retourne
un vecteur 1D de forme `(n_lignes,)`, égal à `predict_proba(X)[:, 1]`.
Ce contrat vaut aussi quand `positive_class=0` ou `positive_class=False`.

```python
# df["status"] contient "retained" et "churn".
result = AutoML(
    target="status", positive_class="churn", models="logistic_regression",
).fit(df)

assert result.positive_class == "churn"
assert result.negative_class == "retained"
# result.classes_ == ["retained", "churn"]
p_churn = result.predict_positive_proba(df.drop(columns="status"))
```

L'estimateur brut `result.best_model` travaille sur les labels internes 0/1.
Les métriques optimisées en CV et calculées sur le test utilisent ce même
encodage : precision, recall et f1 concernent la classe positive explicite ;
roc_auc et average_precision utilisent sa probabilité. log_loss utilise aussi
les labels/probabilités dans cet ordre. `result.label_encoder` est un helper
binaire interne ; son type concret n'est plus sklearn.LabelEncoder. Utiliser
les attributs et méthodes publics du résultat pour l'inférence.

## Analyse du code initial et décisions de migration

L'analyse initiale portait sur les six fichiers présents dans `ml`, qui ne
contenait ni packaging, ni tests, ni dépôt Git. Les imports historiques vers
le package `mlib`, ainsi que la casse des modules Strategy / Metrics, ne
correspondaient à aucun package présent dans ce dossier.

| Fichier initial | Responsabilité et constat |
| --- | --- |
| `strategy.py` | Options, DataFrame, domaines Optuna, conversion de cible en place et discrétisation. |
| `sco_mod.py` | Boucle modèles/métriques, optimisation sur toutes les données, rapports et Styler/Excel. |
| `scoring.py` | Suggestions Optuna, CV, early stopping, pruning et choix des seuils mélangés ; API obsolètes et affectation du résultat de `append`. |
| `report.py` | Split tardif, entraînement, early stopping sur le test, seuils ajustés sur le test, sauvegarde et présentation. |
| `metrics.py` | Métriques usuelles, importances et utilitaires de seuil/aires partielles, avec dépendance Vertica inutilisée. |
| `second_step.py` | Segmentation explicite et orchestration, erreurs avalées et sortie Styler. |

Le nouveau package est ajouté **à côté des modules historiques** pour conserver
leurs imports locaux pendant la migration. Les domaines des sept algorithmes
historiques sont déplacés dans `optimization/spaces.py`, avec copies
indépendantes, puis utilisés par `local_optimizer` et le pont legacy. Les
suggestions modernes utilisent `suggest_float(..., log=True)`.

Les espaces LightGBM/XGBoost reprennent les bornes historiques. Pour la nouvelle
LogisticRegression, l'espace est limité à L2 avec lbfgs, `C` entre 1e-4 et 1e4,
`tol` entre 1e-4 et 1e-2, et 500 à 1000 itérations. Cela évite les combinaisons
solver/penalty incompatibles et l'ancien domaine de C couvrant 17 ordres de
grandeur. Les domaines historiques penalty/l1_ratio sont conservés dans les
métadonnées pour une migration ultérieure ; le pont logistic legacy utilise
également L2, avec lbfgs ou saga et les bornes C/tol/max_iter historiques.

## Isolation du test et contenu du résultat

```text
DataFrame validé et copié
    ├── TEST stratifié, réservé
    └── TRAIN
          ├── Optuna + StratifiedKFold
          ├── meilleurs paramètres et sélection du modèle par score CV
          └── entraînement final sur tout TRAIN
                    └── évaluation unique sur TEST
```

Le test n'entre jamais dans l'optimiseur, la sélection des paramètres/modèles,
le choix d'un seuil ou l'early stopping. La V0.1 désactive l'early stopping et
le pruning, et utilise les prédictions natives des estimateurs. Seul le gagnant
CV est réentraîné et évalué sur le test ; les autres modèles restent dans le
leaderboard CV.

- `leaderboard` : DataFrame indexé par modèle, avec rank, cv_score et cv_std,
  trié suivant maximize/minimize. Les ex æquo conservent l'ordre des modèles.
- `cv_results` : DataFrame brut avec modèle, numéro d'essai, score moyen et
  score de chaque fold, pour tous les essais terminés.
- `best_params` : paramètres effectifs de l'estimateur final, dont le seed.
- `test_metrics` : les huit métriques binaires du seul gagnant ; `metrics`
  en contient une copie pour l'accès générique.
- `feature_importance` : DataFrame numérique ; importances natives des arbres
  ou coefficients signés de LogisticRegression, triés par valeur absolue.
- `config` : copie validée de la configuration ; `train_indices` et
  `test_indices` sont les positions des lignes, même avec un index dupliqué.
- `positive_class`, `negative_class` et `classes_` : contrat de labels et de
  colonnes de probabilités décrit ci-dessus.

Les essais Optuna sont séquentiels et leur sampler est seedé. `n_jobs` vaut -1
ou un entier positif, et contrôle les threads des modèles, avec threadpoolctl
pour le calcul numérique. `n_trials` et `timeout` s'appliquent **par modèle**.
Un timeout dépend du temps réel : il peut changer le nombre d'essais terminés,
même avec un seed identique. `verbose` contrôle les messages du logger
`rationalml.automl`, sans modifier la configuration globale d'Optuna.

## Arborescence et responsabilités

```text
ml/
├── .gitignore
├── pyproject.toml
├── README.md
├── rationalml/
│   ├── __init__.py              # exports publics
│   ├── automl.py                # orchestration du cycle ML
│   ├── config.py                # options et validation
│   ├── data.py                  # validation et encodage binaire explicite
│   ├── result.py                # résultats bruts et inférence
│   ├── exceptions.py            # erreurs explicites
│   ├── legacy.py                # adaptation des sept noms historiques
│   ├── tasks/
│   │   ├── __init__.py          # exports des tâches
│   │   └── base.py              # Enum, normalisation et résolution binaire
│   ├── models/
│   │   ├── __init__.py          # exports modèles
│   │   ├── base.py              # ModelSpec et importances
│   │   └── registry.py          # enregistrement et chargement des modèles
│   ├── evaluation/
│   │   ├── __init__.py          # exports évaluation
│   │   ├── metrics.py           # MetricSpec, dispatch et évaluation
│   │   └── registry.py          # registre des huit métriques binaires
│   └── optimization/
│       ├── __init__.py          # namespace optimisation
│       ├── optimizer.py         # Optuna et CV sur train uniquement
│       └── spaces.py            # espaces adaptés et domaines historiques
├── tests/
│   ├── conftest.py              # petit dataset make_classification
│   ├── test_config_validation.py
│   ├── test_task_type.py
│   ├── test_metric_registry.py
│   ├── test_model_registry.py
│   ├── test_binary_automl.py
│   ├── test_legacy_compatibility.py
│   ├── test_positive_class.py
│   └── test_legacy_partial_areas.py
├── strategy.py                 # legacy conservé et corrigé
├── sco_mod.py                  # pont vers le nouveau moteur
├── scoring.py                  # contrat train-only explicite
├── report.py                   # anciens helpers retirés explicitement
├── metrics.py                  # utilitaires historiques conservés
└── second_step.py              # segmentation legacy sans erreurs masquées
```

Le cœur n'importe aucun de ces six modules legacy. Le chemin historique
`local_optimizer` → `sco_mod` → nouveau moteur reste utilisable, de même que
`second_step`. `scoring.opti` reste disponible pour un train déjà isolé, et
`Metrics` conserve ses utilitaires, avec les aires partielles retirées ci-dessous.
`report.py` reste importable, mais ses anciens
helpers d'entraînement/reporting lèvent une erreur de migration explicite.

Pour ajouter une métrique ou un modèle : créer un `MetricSpec` / `ModelSpec`,
puis appeler `MetricRegistry.register` / `ModelRegistry.register`. Le moteur
reste inchangé. Un modèle enregistré doit accepter random_state et déclarer
son paramètre de threads via `n_jobs_parameter` (ou None). La V0.1 requiert
predict_proba pour les huit métriques finales.

## Retrait des aires partielles legacy

La V0.1.1 applique l'option B : **`Metrics.sub_area` et `Metrics.sub_area_pr`
lèvent systématiquement `LegacyAPIError`**. Leurs noms et signatures restent
présents pour fournir une erreur de migration explicite.

L'audit de `metrics.py` confirme que `sub_area` détruisait l'information des
probabilités en les convertissant en 0/1 avant de reconstruire la courbe ROC.
`sub_area_pr` pouvait intégrer un intervalle entier au-delà de la borne de
recall, sans interpolation à cette borne. Aucun contrat suffisamment précis
ne définit la borne métier, l'interpolation et la normalisation souhaitées.
Les calculs incorrects ont donc été supprimés, sans remplacement implicite.

Les métriques enregistrées `roc_auc` et `average_precision` restent disponibles
pour une évaluation globale ; elles ne remplacent pas une aire partielle
équivalente. Un retour des aires partielles nécessitera un contrat métier
précis et des tests mathématiques dédiés.

## Changements incompatibles explicites

Le projet s'appelle **RationalML**, et le package Python s'importe avec
`from rationalml import AutoML`. Les scripts clients doivent mettre à jour
leurs imports vers `rationalml`. La version reste 0.1.1. Pour installer les
extras à partir d'une distribution publiée, le nom est `RationalML[boosting]` ;
l'installation locale décrite ci-dessus utilise le même projet.

Ajouts V0.1.1 :

- Les cibles non standards sans `positive_class` émettent maintenant un
  `UserWarning`. Leur choix positif par défaut reste celui de V0.1.
- Une classe positive explicite peut changer l'ordre des colonnes de
  `predict_proba` et le sens des métriques par rapport à V0.1. Utiliser
  `classes_` ou `predict_positive_proba` pour connaître P(classe positive).
- Les appels à `Metrics.sub_area` / `sub_area_pr` échouent explicitement avec
  `LegacyAPIError`, au lieu de produire un nombre incorrect.

Ruptures de la migration V0.1 toujours applicables :

- `sco_mod` et `second_step` retournent un DataFrame brut, avec un schéma de
  leaderboard CV, au lieu d'un Styler et des anciennes colonnes de reporting.
  Les `AutoMLResult` legacy sont disponibles dans `init_model.results_` et
  `init_model.result_` pour un appel direct à sco_mod.
- Les options legacy de seuil, pruning et early stopping sont retirées du
  moteur et produisent un avertissement à chaque appel à sco_mod. Les
  métriques de classification utilisent les prédictions natives.
- `excel_report`, `model_save`, `model_select` et `class_weight="auto"` du
  chemin legacy produisent `LegacyAPIError`. Le calcul historique automatique
  des poids utilisait la cible entière. Les poids numériques fixes restent
  utilisables dans le pont legacy.
- `scoring.opti` requiert `training_only=True`, avec labels train 0/1. Sa
  seconde sortie contient cv_std ; les anciennes variances roc/pr sont retirées.
  `scoring.objective`, `report.prediction`, `prediction_next` et `report_global`
  lèvent `LegacyAPIError`, car leurs contrats ne prouvent pas l'isolation du test.
- `_pandas_sets` et `discretize` travaillent désormais sur une copie. Les
  appels à discretize doivent récupérer sa valeur de retour.
- Les imports sont `rationalml` pour la nouvelle API et les noms locaux
  en minuscules pour les modules historiques de ce dossier. Aucun faux package
  `mlib` n'est créé.

## Vérification et suite de migration

Vérification V0.1.1 : **129 tests réussis en 3,45 s**, avec **un avertissement
attendu** dans le test V0.1 utilisant des chaînes sans `positive_class`.
Les 98 tests V0.1 sont conservés sans modification ; 31 cas supplémentaires
couvrent le contrat de classe positive et le retrait des aires partielles.
Installation éditable, imports depuis le dossier parent et `pip check` réussis.
Environnement testé : Python 3.12.6, pandas 3.0.6, numpy 2.5.3,
scikit-learn 1.9.1, Optuna 5.0.0, LightGBM 4.7.0 et XGBoost 3.4.1.

La suite couvre validation, tâches, registres extensibles, fit/inférence,
labels originaux, absence de mutation, log_loss minimisée, reproductibilité
des trois modèles et fonctionnement des sept alias legacy. Les tests de fuite
tracent les lignes vues par Optuna, chaque fit et chaque prédiction ; ils
vérifient l'évaluation unique du test. Un autre test perturbe uniquement les
features du test et exige des hyperparamètres, un gagnant et des scores CV
inchangés. Les erreurs d'entraînement et des segments remontent au caller.

Les nouveaux tests vérifient `churn` encodé en 1 pour les trois modèles,
l'inférence et les colonnes de probabilités, les six métriques liées aux labels
en CV et sur test, les classes explicites absentes, les choix par défaut 0/1 et
booléens, les choix explicites 0/False, les avertissements déterministes et
`LegacyAPIError` pour les deux aires partielles.

Avant V0.2 : définir la migration des options et schémas legacy retirés,
préciser le contrat métier d'un éventuel retour des aires partielles et élargir
la matrice CI aux versions de dépendances annoncées. Un éventuel retour de l'early stopping devra
utiliser une validation issue du train exclusivement. Multiclass, régression,
MLflow, Excel, SHAP, calibration, ranking/déciles et preprocessing avancé
restent hors de cette étape.
