# RationalML — V0.7.0

RationalML compare et optimise des modèles de classification binaire,
multiclass et de régression sur les
variables choisies par le Data Scientist. **RationalML n'est pas un outil de
feature engineering automatique. Il suppose par défaut que le Data Scientist
fournit un DataFrame prêt pour la modélisation.**

Le Data Scientist garde la main sur le nettoyage, le choix des variables,
le feature engineering et les transformations métier. RationalML prend en
charge le split train/test, la cross-validation, Optuna, la comparaison des
modèles, les métriques et les résultats.

| `preprocessing` | Contrat |
| --- | --- |
| `None` **(défaut)** | Contrôle total du Data Scientist : aucune transformation des features |
| `"basic"` | Commodité minimale : imputation, OneHotEncoding et scaling selon le modèle |
| Transformer sklearn | Preprocessing utilisateur, cloné et ajusté à l'intérieur de chaque fold CV |

## Installation et utilisation principale

Python 3.10 ou plus, depuis ce dossier :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[boosting,legacy,test,excel,mlflow]"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

Le cœur requiert numpy, pandas, scikit-learn, Optuna et threadpoolctl. LightGBM
et XGBoost sont optionnels via l'extra `boosting`. `models="auto"` sélectionne
les modèles enregistrés dont les dépendances sont installées ; une dépendance
explicitement demandée mais absente produit `MissingDependencyError`.
L'extra `legacy` fournit tqdm pour `local_optimizer`. Les modèles sont filtrés
par tâche : les régresseurs ne sont jamais proposés à une classification.

```python
from rationalml import AutoML

# df_model : features déjà préparées par le Data Scientist, cible churn 0/1.
result = AutoML(
    target="churn",
    positive_class=1,
    preprocessing=None,  # défaut
    models="auto",
    metric="roc_auc",
).fit(df_model)

predictions = result.predict(new_df_model)
probabilities = result.predict_proba(new_df_model)
p_churn = result.predict_positive_proba(new_df_model)
```

Les autres tâches utilisent également un DataFrame préparé, sans preprocessing
par défaut :

```python
multiclass_result = AutoML(
    target="segment", task="multiclass",
).fit(df_multiclass)
segments = multiclass_result.predict(new_df_multiclass)
segment_probabilities = multiclass_result.predict_proba(new_df_multiclass)
# Colonne j = P(segment == multiclass_result.classes_[j]).

regression_result = AutoML(
    target="revenue", task="regression",
).fit(df_regression)
revenues = regression_result.predict(new_df_regression)
```

| Tâche | `metric="auto"` (défaut) | CV |
| --- | --- | --- |
| `binary` | `roc_auc` | `StratifiedKFold` |
| `multiclass` | `f1_macro` | `StratifiedKFold` |
| `regression` | `rmse` | `KFold` |

Les deux splitters utilisent `shuffle=True` et `random_state`. Le split
train/test est stratifié seulement pour la classification.
Avec `task="auto"`, exactement deux valeurs donnent binary ; plus de deux
labels non numériques donnent multiclass. **Numeric targets with more than
two values require an explicit task.** L'ambiguïté produit
`UnsupportedTaskError`, sans heuristique de cardinalité. Une cible à une seule
classe exige notamment `task="regression"` si c'est une mesure constante.

| Tâche | Modèles | Métriques disponibles |
| --- | --- | --- |
| binary | `logistic_regression`, `lightgbm`, `xgboost` | roc_auc, average_precision, accuracy, balanced_accuracy, precision, recall, f1, log_loss |
| multiclass | `logistic_regression`, `lightgbm`, `xgboost` | accuracy, balanced_accuracy, f1_macro, f1_weighted, log_loss |
| regression | `ridge`, `lightgbm_regressor`, `xgboost_regressor` | rmse, mae, r2 |

LightGBM/XGBoost restent optionnels. `log_loss`, `rmse` et `mae` sont minimisées ;
les autres métriques sont maximisées. RMSE utilise
[`root_mean_squared_error`](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_error.html),
disponible dans les versions sklearn supportées. Une métrique incompatible
échoue avant Optuna. `result.primary_metric` et `result.config.metric` exposent
le nom effectif ; `result.config.task` expose la tâche résolue.

Avec `None`, les features doivent être numériques ou booléennes, sans NaN
ni infini. RationalML n'impute, n'encode, ne scale et ne supprime aucune
colonne, même pour LogisticRegression. Des catégories ou NaN produisent
`DataValidationError`, avec les trois choix : préparer les données avant
RationalML, utiliser `"basic"` ou fournir un transformer sklearn.

## Option de convenance : basic

Le mode `"basic"` est **« basic preprocessing, not feature engineering »**.
Il reprend uniquement les transformations de V0.2.0, sur demande explicite.
Par exemple, `df_raw` peut contenir age, income, country, is_customer et churn,
avec des catégories et des valeurs manquantes :

```python
basic_result = AutoML(
    target="churn", positive_class=1, preprocessing="basic",
    models="auto", metric="roc_auc",
).fit(df_raw)

basic_result.predict(new_df_raw)
basic_result.predict_positive_proba(new_df_raw)
```

| Type brut | Transformation basic |
| --- | --- |
| Entiers et flottants, y compris dtypes nullable | `SimpleImputer(strategy="median")`, puis scaling selon le modèle |
| `bool` / `boolean` nullable | Représentation 0/1 et imputation par le mode, sans scaling |
| `object`, `category`, `string` | Normalisation des manquants, imputation par le mode et `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` |
| Datetime, datetime avec timezone, timedelta | `DataValidationError` explicite |

En mode basic, `ModelSpec.requires_scaling=True` active `StandardScaler` pour
LogisticRegression et Ridge ; LightGBM/XGBoost déclarent False et ne sont pas scalés.
Ce champ n'a aucun effet avec `None` ou un transformer utilisateur.

`FeatureSchema` est inféré uniquement sur les lignes d'entraînement du fold,
puis sur le train complet du pipeline final. Les constantes sont conservées
avec `UserWarning`. Une colonne entièrement manquante dans un train, y
compris un fold, produit `DataValidationError` : aucun fold n'emprunte de
statistique aux autres lignes. Les infinis et structures imbriquées sont
rejetés ; aucune ligne ou colonne n'est supprimée silencieusement.

Les catégories doivent être des scalaires homogènes : chaînes ou nombres,
sans mélange des deux familles. Les catégories pandas non observées dans
le train ne sont pas apprises depuis les métadonnées du dtype. Une catégorie
inconnue à l'inférence devient un bloc OneHot de zéros. Aucun `LabelEncoder`
n'est appliqué aux features.

`PreprocessingConfig` reste un réglage avancé facultatif du mode basic,
avec les mêmes cinq options qu'en V0.2.0, sans ajout :

```python
from rationalml import AutoMLConfig, PreprocessingConfig

config = AutoMLConfig(
    target="churn", preprocessing=PreprocessingConfig(scale_numeric=False),
)
```

Ses défauts sont median / most_frequent / onehot / scale_numeric="auto" /
handle_unknown="ignore". Le numérique accepte aussi mean et most_frequent,
le scaling accepte True/False, et handle_unknown accepte "error".

## Option expert : transformer sklearn utilisateur

Le Data Scientist fournit un transformer compatible sklearn, avec `fit`,
`transform`, `get_params`, et clonable par `sklearn.base.clone`.
Un `Pipeline`, un `ColumnTransformer` ou un transformer personnalisé convient.
RationalML n'ajoute aucune transformation autour de cet objet.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Choix explicites du Data Scientist. country contient des chaînes ou None.
my_column_transformer = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), ["age", "income"]),
    ("country", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), ["country"]),
])

expert_result = AutoML(
    target="churn", positive_class=1, preprocessing=my_column_transformer,
    models="logistic_regression", metric="roc_auc",
).fit(df_raw)

expert_result.predict(new_df_raw)
expert_result.predict_proba(new_df_raw)
```

Adapter l'imputer aux données fournies : l'exemple utilise None pour les
catégories, plutôt que NaN, défaut sklearn. Le transformer est responsable
de ses dtypes, valeurs et transformations. RationalML vérifie seulement la
structure et les colonnes brutes dans ce mode.

**À chaque fold**, RationalML appelle `clone(user_transformer)`, construit un
Pipeline neuf avec un estimateur neuf, puis ajuste le tout sur le fold train.
Le fold validation est seulement transformé. Pour le fit final, RationalML
clone encore le transformer et l'ajuste sur le train complet, jamais sur le
holdout. L'objet fourni n'est pas ajusté ni modifié par RationalML ; le
clonage sklearn standard ne reprend pas son éventuel état déjà appris.
Un transformer dont le clonage renvoie la même instance est rejeté.
Voir le [contrat de clone](https://scikit-learn.org/1.5/modules/generated/sklearn.base.clone.html).

Les transformations réalisées en amont restent sous la responsabilité du
Data Scientist. Pour couvrir par la CV les transformations apprenant des
statistiques, fournir leur transformer dans `preprocessing`.

## Validation et inférence

`fit` exige un DataFrame avec des noms de colonnes uniques, non vides et de
type chaîne, et une cible sans valeur manquante. Binary exige exactement deux
classes, multiclass au moins trois, et regression une cible numérique finie.
Chaque classe doit conserver au moins `cv` observations dans le train ; toutes
les classes doivent être présentes dans le test. Les catégories pandas non
observées de la cible ne comptent pas comme classes. En régression, au moins
deux lignes par fold validation et dans le test sont exigées pour que R2 soit
défini. Aucune cible n'est imputée.

Dans les trois modes, un DataFrame d'inférence doit contenir exactement les
mêmes noms de features, chacun une fois. Leur ordre peut changer : RationalML
les réordonne selon `feature_names`. Les colonnes manquantes, supplémentaires
ou dupliquées provoquent `DataValidationError`. L'entrée est copiée avant son
utilisation par le pipeline.

- `None` : données prêtes, numériques/booléennes, sans NaN ni infini ; aucune
  inférence de schéma, recherche de constantes ou transformation automatique.
- `"basic"` : contrôles de types V0.2 ; int/float compatibles, booléens ou 0/1,
  catégories de même famille. Les NaN et catégories inconnues sont acceptés,
  y compris un batch entièrement manquant, grâce au preprocessing ajusté.
- Transformer : données brutes transmises sans les règles de `FeatureSchema`
  basic ; le propre contrat de dtypes du transformer fait foi.

Une matrice numpy de même largeur est reconstruite avec les noms dans l'ordre
d'origine. Le mode basic exige un DataFrame si des catégories sont présentes.
Le DataFrame reste recommandé pour le mode expert et ses sélections par nom.

## Targets et résultats

Le contrat V0.1.1 reste inchangé. `positive_class` explicite est vérifiée
parmi les deux labels puis encodée en 1 ; l'autre classe devient 0.
Sans choix explicite : booléens → True, cible exactement {0, 1} → 1, autres
labels comparables → plus grand label selon l'ordre naturel avec `UserWarning`
précisant le choix et recommandant `positive_class`.
Le choix est déterministe, indépendant des fréquences et de l'ordre des lignes.
Une classe positive absente produit `DataValidationError` ; 0, False et une
chaîne vide restent des choix explicites valides lorsqu'ils sont présents.

`AutoMLResult` expose `positive_class`, `negative_class` et `classes_` dans
l'ordre **[negative_class, positive_class]**. `predict` restitue les labels
originaux. `predict_proba` retourne P(negative_class) en colonne 0 et
P(positive_class) en colonne 1 ; `predict_positive_proba` retourne ce dernier
vecteur 1D. Precision, recall, f1, roc_auc et average_precision utilisent la
classe interne 1, y compris si le label original positif est 0 ou False.

En multiclass, un `MulticlassLabelEncoder` léger encapsule sklearn LabelEncoder,
ajusté uniquement sur le train. Il encode les labels en 0…n_classes-1 dans
l'ordre déterministe de `classes_`, rejette les labels inconnus et restaure
les labels originaux dans `predict`. `predict_proba` renvoie la matrice complète,
colonne j correspondant à `classes_[j]`, également utilisée par log_loss.

En régression, aucun encodeur : y conserve son dtype numérique et `predict`
renvoie les valeurs numériques. `classes_` et `predict_proba` lèvent
`UnsupportedTaskError`. `positive_class`, `negative_class` et
`predict_positive_proba` lèvent cette erreur hors binary. Renseigner
`positive_class` hors binary produit `ConfigurationError` après résolution
de la tâche.

- `task`, `target`, `config`, `primary_metric` : tâche résolue, cible, copie
  validée des options et métrique effective.
- `leaderboard` : scores et diagnostics CV, triés selon la direction de la métrique.
- `cv_results` : modèle, essai, score moyen et scores des folds Optuna.
- `best_model_name`, `best_params` : gagnant CV et paramètres de son estimateur.
- `best_model` : **`sklearn.pipeline.Pipeline`**. Avec None : étape `estimator`
  seule. Avec basic ou transformer : `preprocessing`, puis `estimator`.
  Les attributs natifs sont dans `best_model.named_steps["estimator"]` ;
  ses labels de classification restent internes. Utiliser `AutoMLResult` pour les labels
  originaux et les contrôles de colonnes.
- `metrics`, `test_metrics` : toutes les métriques de la tâche du seul gagnant,
  calculées lors d'une évaluation unique sur le test.
- `feature_names` : noms bruts dans l'ordre d'entraînement.
- `feature_schema` : schéma du train final en mode basic ; None ailleurs.
- `transformed_feature_names` : noms fournis au modèle, ou None si indisponibles.
- `feature_importance` : DataFrame brut, colonnes `feature`, `importance`
  numérique et `source_feature` en binary/régression. Coefficients logistiques
  et Ridge signés, importances natives des arbres, triés par valeur absolue.
  LogisticRegression multiclass ajoute `class` : une ligne par classe et feature,
  sans moyenne ni perte de lignes de coef_. Les arbres multiclass conservent
  leur importance native globale par feature. Aucun Styler.
- `train_indices`, `test_indices` : positions des lignes, même avec index dupliqué.

Avec None, les noms et source_feature sont ceux d'origine. Avec basic,
`ColumnTransformer.get_feature_names_out()` fournit par exemple
`categorical__country_France`. Le mapping source utilise les slices de sortie
et les catégories apprises, sans découper les noms sur des underscores.
Les caractères spéciaux sont échappés pour LightGBM/XGBoost.

Avec un transformer utilisateur, RationalML essaie `get_feature_names_out`
et vérifie la cohérence des noms. Si les noms sont indisponibles ou non fiables,
`transformed_feature_names=None` et `feature_importance` est un DataFrame vide
avec `UserWarning`. Le fit et l'inférence restent utilisables. Quand les noms
sont disponibles, `source_feature` reste manquant : aucune filiation n'est
inventée pour les transformations utilisateur.

## Baseline et stabilité CV

Une baseline naïve est évaluée sur **TRAIN uniquement**, avec la même métrique
principale et exactement les mêmes folds que les modèles :
[`DummyClassifier(strategy="prior")`](https://scikit-learn.org/1.5/modules/generated/sklearn.dummy.DummyClassifier.html)
en binary/multiclass, et
[`DummyRegressor(strategy="mean")`](https://scikit-learn.org/1.5/modules/generated/sklearn.dummy.DummyRegressor.html)
en regression. Elle apprend seulement la cible, sans preprocessing des features
ni Optuna. Elle reste un diagnostic, hors candidats à `best_model`, sans refit
sur le train complet ni métrique calculée sur TEST.

```python
result.baseline_name         # "dummy_classifier" ou "dummy_regressor"
result.baseline_score        # moyenne des scores des folds
result.baseline_cv_std       # np.std(...), ddof=0
result.baseline_fold_scores  # tuple des scores, dans l'ordre des folds
result.leaderboard
# rank, cv_score, cv_std, cv_min, cv_max, improvement_vs_baseline, n_trials_completed
```

`cv_std`, `cv_min` et `cv_max` décrivent les folds du meilleur trial de chaque
modèle, sans score ni seuil arbitraire de stabilité. La convention de l'écart
type reste `ddof=0`. `improvement_vs_baseline` vaut modèle − baseline pour
maximize, baseline − modèle pour minimize : ROC-AUC 0,8 contre 0,5 donne +0,3 ;
RMSE 8 contre 10 donne +2. Une valeur négative indique une performance inférieure
à la référence. Le classement et le gagnant restent fondés uniquement sur
`cv_score`. `n_trials_completed` compte seulement les essais Optuna `COMPLETE`,
utile avec timeout ; FAIL et PRUNED sont exclus. Aucun timing n'est ajouté.

## Business ranking

**All ranking diagnostics are computed on the untouched holdout test set.**
Les analyses lisent les prédictions de l'évaluation initiale, après sélection
et refit du gagnant. Aucun nouvel appel au modèle ; aucun retour vers Optuna,
la baseline ou le preprocessing. `result.test_predictions` retourne une copie
indépendante avec l'index original, y compris ses doublons :

| Tâche | Colonnes de `test_predictions` |
| --- | --- |
| binary | `y_true`, `y_pred` en labels originaux ; `score` = P(positive_class) |
| multiclass | `y_true`, `y_pred` en labels originaux ; `proba_0`, `proba_1`… ; colonne `proba_j` = P(classes_[j]) |
| regression | `y_true`, `y_pred`, `residual` = y_true − y_pred |

Les noms de probabilités utilisent leur position, sans dépendre du texte des
labels. L'inférence ultérieure ne modifie pas cet artefact.

```python
# Binary : score de la classe positive déjà définie.
ranking = result.ranking_table(n_bins=10)
ranking[["segment", "count", "positive_rate", "cumulative_positive_capture", "lift"]]
top10 = result.top_segment(0.10)
top10.score.mean()

# Multiclass : analyse one-vs-rest explicite, class_label obligatoire.
multiclass_result.ranking_table(class_label="gold", n_bins=10)
multiclass_result.top_segment(0.10, class_label="gold")

# Regression : niveaux de prédiction, sans notion de classe positive ou lift.
regression_result.ranking_table(n_bins=10)
regression_result.top_segment(0.10)
```

Le segment 1 contient les scores/prédictions les plus élevés. Le tri décroissant
est stable : les égalités gardent l'ordre stocké du holdout. Le découpage utilise
la position, avec `effective_bins=min(n_bins, n_rows)` : groupes non vides,
tailles différant d'au plus une ligne, sans qcut. `n_bins` doit être un entier
>= 2, hors bool. Le top contient exactement `ceil(n_rows*fraction)` lignes,
au moins une, sans extension pour les égalités ; `0 < fraction <= 1`.

Binary/one-vs-rest retourne `segment`, `count`, `score_min`, `score_max`,
`score_mean`, `positives`, `positive_rate`, `population_share`, `positive_capture`,
`cumulative_positive_capture`, `lift`, `cumulative_lift`. La capture rapporte les
positifs aux positifs totaux ; le lift rapporte le taux positif au taux global,
et sa version cumulative utilise les lignes depuis le segment 1. Zéro positif
produit une erreur explicite. `class_label` est réservé au multiclass et vérifié
parmi `classes_`.

Regression retourne `segment`, `count`, `pred_min`, `pred_max`, `pred_mean`,
`actual_mean`, `bias`, `mae`, `rmse`. Le biais vaut pred_mean − actual_mean,
donc l'opposé du résidu moyen. Tous les tableaux contiennent des valeurs brutes,
sans Styler, formatage ou seuil interprétatif.

## Excel export

L'extra Excel est optionnel ; le cœur s'importe et fonctionne sans openpyxl :

```shell
pip install "rationalml[excel]"
# Depuis ce dépôt : python -m pip install -e ".[excel]"
```

```python
result.to_excel("model_report.xlsx")  # retourne un pathlib.Path

from rationalml import ExcelReportConfig

config = ExcelReportConfig(include_predictions=True, top_fraction=0.10)
result.to_excel("model_report.xlsx", config=config)

# Multiclass : classe métier choisie explicitement.
multiclass_result.to_excel("segment_report.xlsx", class_label="gold")
```

Les feuilles communes sont **Summary**, **Leaderboard**, **Metrics**,
**Feature Importance**, **Hyperparameters**. Binary et regression ajoutent
**Ranking** et **Top Segment**. En multiclass, ces deux feuilles sont présentes
uniquement si `class_label` est fourni ; sans classe, le reste du rapport est
exporté normalement. **Predictions** est créée seulement avec
`include_predictions=True` (False par défaut), à partir du holdout stocké.
Top Segment et Predictions conservent l'index dans leur première colonne,
sans exporter les features d'origine. Les index composites sont représentés
en JSON et les dates avec fuseau en ISO, pour conserver leurs identifiants.

`ExcelReportConfig` est une dataclass frozen : `decimal_places=3`,
`percentage_places=2` (entiers entre 0 et 15), `include_predictions=False`,
`top_fraction=0.10` avec `0 < top_fraction <= 1`. Les nombres restent numériques :
formats `0.000` pour les décimales, `0.00%` pour les taux/captures, `0.00` pour
les lifts et `0` pour les entiers. La locale est gérée par Excel. Le style
reste sobre : en-têtes gras, filtres, première ligne figée et largeurs bornées.

L'export lit uniquement AutoMLResult et appelle ses méthodes ranking/top
existantes ; aucun fit, predict, predict_proba, recalcul des métriques ou
appel Optuna. Une importance indisponible n'empêche pas l'export.
Openpyxl est importé à la demande ; s'il manque, `MissingDependencyError`
indique la commande d'installation. L'extra requiert `openpyxl>=3.0.10,<4` ;
l'écriture directe ne dépend pas des exigences du writer Excel de pandas.

Le chemin doit avoir l'extension `.xlsx` et un parent existant. Les dossiers
ne sont pas créés et un fichier existant est écrasé. Les limites de
1 048 576 lignes **en-tête compris** et 16 384 colonnes sont vérifiées avant
sauvegarde : aucune troncature silencieuse. Les textes trop longs ou les
nombres infinis provoquent aussi une erreur explicite.

## MLflow tracking

MLflow est une couche optionnelle de traçabilité **après fit**, sans connexion
au moteur d'optimisation :

```shell
pip install "rationalml[mlflow]"
# Depuis ce dépôt : python -m pip install -e ".[mlflow]"
```

```python
from rationalml import MLflowConfig

run_id = result.log_mlflow(
    config=MLflowConfig(
        experiment_name="customer-churn",
        run_name="churn-model",
        tracking_uri="http://127.0.0.1:5000",
    )
)

# Multiclass : ranking seulement pour la classe explicitement choisie.
multiclass_result.log_mlflow(
    config=MLflowConfig(experiment_name="segments", log_model=False),
    class_label="gold",
)
```

Chaque appel crée **un run RationalML et un enfant par modèle du leaderboard**,
jamais un run par trial Optuna. Le parent contient les métadonnées explicites
du fit, la baseline, le meilleur score CV et les métriques test préfixées
`test_`. Chaque enfant conserve ses meilleurs paramètres dans
`result.model_best_params`, ses diagnostics CV et le tag `rationalml.selected`.

Les tables sont `leaderboard.json` (modèle en colonne), `cv_results.json`,
`feature_importance.json` si disponible et `ranking.json` en binary/regression.
En multiclass, ranking nécessite `class_label` ; une classe invalide est
rejetée avant création des runs et la classe choisie est tracée dans le
paramètre parent `ranking_class`. Aucun Top Segment n'est envoyé.
`log_predictions=False` par défaut : aucune observation individuelle du
holdout. L'opt-in ajoute `predictions.json` avec une première colonne d'index
original, renommée par suffixe si son nom entre en conflit. Les features
originales et datasets train/test ne sont jamais envoyés.

`MLflowConfig` est frozen : `experiment_name="RationalML"`, `run_name=None`,
`tracking_uri=None`, `log_model=True`, `log_predictions=False`, `nested=False`,
`tags=None`. Les tags sont copiés et leurs valeurs scalaires converties en
texte. Les clés `rationalml.*` et `mlflow.parentRunId` sont réservées ; un
conflit produit `ConfigurationError`. Les paramètres structurés utilisent
du JSON trié ; un objet non sérialisable est représenté par `<module.Class>`
avec warning, sans adresse mémoire.

`log_model=True` sérialise uniquement le Pipeline gagnant complet avec l'API
MLflow 3 `name="model"` et cloudpickle, y compris le preprocessing utilisateur.
Aucun input_example, signature inférée, fit, predict, predict_proba, autologging
ou enregistrement au Model Registry. Le résultat reste inchangé ; seul le
`run_id` est retourné.

Une URI fournie est temporaire et restaurée même en cas d'échec ; None respecte
la configuration MLflow existante. Aucun serveur n'est lancé. Un run utilisateur
actif exige `nested=True` et la même URI : il reste actif, tandis que les runs
RationalML sont fermés. L'expérience est sélectionnée/créée via son ID sans
changer l'expérience active de l'utilisateur. Un échec du backend remonte.
MLflow est importé uniquement à la demande ; s'il manque, `MissingDependencyError`
indique l'installation de l'extra `mlflow` (`mlflow>=3.1,<4`).

## Protocole et responsabilités

```text
FULL DATA
    ├── TEST réservé
    └── TRAIN
          ├── encodage cible si classification, création unique des folds
          ├── baseline CV sur ces folds, diagnostic cible uniquement
          ├── CV / Optuna
          │     ├── fold TRAIN : Pipeline neuf, fit sur ces lignes uniquement
          │     └── fold VALIDATION : transform et évaluation
          ├── choix du modèle et des paramètres par score CV
          └── Pipeline final neuf, fit sur TRAIN complet
                    └── transform et évaluation unique sur TEST
                          └── prédictions conservées → ranking/top sur demande
```

Le holdout ne participe à aucun fit, à Optuna, ni à la sélection des modèles.
Le mode basic construit un preprocessor neuf ; le mode expert clone celui
fourni. Aucun transformer fitted n'est partagé entre folds. Early stopping
et pruning restent désactivés, et les métriques utilisent les prédictions
natives des estimateurs.

Optuna utilise des essais séquentiels et un sampler seedé. `n_jobs` contrôle
les threads des modèles et le calcul numérique. `n_trials` et `timeout`
s'appliquent par modèle ; un timeout réel peut modifier le nombre d'essais
terminés malgré un seed identique. `verbose` pilote le logger RationalML.

Les composants existants conservent leurs responsabilités. V0.3 ajoute
seulement un encodeur multiclass, des entrées de registries et des branches
explicites pour le split/CV et l'inférence. `ModelSpec.task_params` déclare les
objectifs multiclass des arbres ; AutoML ne connaît pas les noms des modèles.
Le sous-package `preprocessing` est inchangé par rapport à V0.2.1, sans
transformation supplémentaire.

Un modèle s'ajoute via `ModelRegistry.register(ModelSpec(...))`, une métrique
via `MetricRegistry.register(MetricSpec(...))`. Le modèle doit accepter
random_state, déclarer `n_jobs_parameter` (ou None), supporter predict_proba en classification
et définir `requires_scaling` pour basic. Aucune modification du moteur n'est
nécessaire pour l'enregistrer.

## Migration V0.6.0 → V0.7.0

Les contrats ML/Excel sont conservés. `model_best_params` est ajouté avec
`default_factory=dict`, sans casser les constructions manuelles existantes.
Pour un ancien résultat sans ce champ, le gagnant utilise `best_params` ;
les autres candidats gardent leurs diagnostics CV avec un warning indiquant
que leurs paramètres sont indisponibles. Aucun paramètre n'est inventé.
Installer l'extra `mlflow` uniquement pour utiliser `log_mlflow`.

## Migration V0.5.0 → V0.6.0

Aucun contrat ML existant ne change. `to_excel` et `ExcelReportConfig` sont
ajoutés ; installer l'extra `excel` uniquement pour exporter. Les helpers Excel
legacy restent désactivés : utiliser `AutoMLResult.to_excel`.

## Migration V0.4.0 → V0.5.0

L'API fit/inférence et le leaderboard gardent leurs contrats. Le résultat conserve
désormais les prédictions du holdout : mémoire proportionnelle à ses lignes
et, en multiclass, au nombre de classes. Les constructions manuelles doivent
renseigner `_test_predictions` ; les anciens résultats sérialisés doivent être
réentraînés pour disposer des nouveaux diagnostics. `test_predictions` est une
propriété sans setter : modifier la copie retournée ne modifie pas le résultat.

## Migration V0.3.0 → V0.4.0

Le leaderboard conserve ses trois colonnes et en ajoute quatre. Les consommateurs
exigeant une liste exacte de colonnes doivent adapter leur sélection.
`AutoMLResult` ajoute les quatre champs baseline ; une construction manuelle
doit les renseigner. `cv_results`, les métriques, les search spaces, la sélection
et les trois modes preprocessing gardent leurs contrats V0.3.

## Migration V0.2.1 → V0.3.0

- `metric` passe de `"roc_auc"` à `"auto"` ; le comportement binaire effectif
  reste roc_auc. Les configurations retournées exposent les noms résolus.
- Les labels textuels à plus de deux classes peuvent désormais résoudre
  multiclass ; les cibles numériques à plus de deux valeurs restent une
  erreur avec auto et nécessitent un choix explicite.
- `positive_class` et les méthodes binaires ne s'étendent pas aux nouvelles
  tâches ; aucune classe positive multiclass n'est inventée.
- `ModelRegistry.available()` / `MetricRegistry.available()` sans filtre
  incluent les nouvelles entrées : passer une tâche pour filtrer.
- L'importance logistique multiclass est un format long avec colonne `class`.
  Les formats binaires, les trois modes preprocessing et les erreurs legacy
  restent conservés.

## Migration V0.2.0 → V0.2.1

- Le défaut est désormais `preprocessing=None` : le DataFrame doit être prêt.
  Utiliser explicitement `"basic"` pour retrouver les transformations V0.2.0.
- `preprocessing="auto"` est retiré avec `ConfigurationError` et indication
  de migration. `models="auto"`, `task="auto"` et le réglage avancé
  `scale_numeric="auto"` restent valides.
- Le pipeline None ne contient plus d'étape `preprocessing`. Les accès directs
  à `best_model.named_steps["preprocessing"]` doivent être conditionnels.
- `feature_schema` est None hors basic. Les résultats numériques peuvent
  changer si le scaling automatique V0.2.0 n'est plus demandé.
- Un transformer sans noms exploitables produit une importance indisponible
  avec warning, sans empêcher l'entraînement.

## Compatibilité legacy

Le package s'importe avec `from rationalml import AutoML`. Les six modules
historiques restent binaires et séparés du cœur. `local_optimizer` → `sco_mod` et
`second_step` utilisent des DataFrames numériques/booléens sans preprocessing
automatique. Leurs résultats sont des DataFrames bruts ; les AutoMLResult
sont disponibles dans `init_model.results_` / `result_`.

Les seuils, pruning et early stopping legacy restent retirés avec warning.
Excel, model_save, model_select et class_weight="auto" du chemin historique
lèvent `LegacyAPIError`. Les poids fixes restent utilisables.
`scoring.opti` exige `training_only=True` et des labels train 0/1 ; sa seconde
sortie contient cv_std. Les anciens helpers `scoring.objective` et ceux de
`report.py` conservent leurs erreurs de migration explicites.

`Metrics.sub_area` et `Metrics.sub_area_pr` restent retirées avec
`LegacyAPIError` depuis V0.1.1 : les anciens calculs ROC/PR partiels étaient
incorrects et leur contrat métier ambigu. `roc_auc` et `average_precision`
évaluent les courbes globales sans prétendre remplacer ces aires partielles.

## Vérification et limites

Vérification V0.7.0 : **689 tests réussis en 249,84 s**, dont **104 nouveaux cas**,
**2 warnings** (classe positive implicite et dépréciation SQLAlchemy `noload`
dans le store SQL MLflow),
**0 échec et 0 skip**. LightGBM et XGBoost sont installés et testés.
`pip check` retourne `No broken requirements found.` avant puis après installation
de l'extra MLflow ; les versions API et installation éditable sont `RationalML==0.7.0`.
Openpyxl **3.0.10**, version minimale déclarée, est testé par écriture directe.
MLflow **3.17.0** est testé avec SQLite et des artefacts locaux sous `tmp_path`,
sans serveur et avec les connexions réseau bloquées dans les tests.

Les 585 cas V0.6 sont conservés sans modification. Les nouveaux tests MLflow
relisent les runs, leurs tables et le Pipeline sérialisé ; ils vérifient
les paramètres par candidat, exactement quatre runs pour trois modèles et
cinq trials, l'opt-in des prédictions et le choix multiclass. Fit/predict,
Optuna, métriques et transformations sont bloqués pendant le tracking dans
les trois modes preprocessing et tâches, avec ou sans modèle. Les tests
couvrent l'import lazy, les runs utilisateurs, les échecs backend, la restauration
des URI/variables d'environnement et l'immutabilité. Les tests Excel relisent
les classeurs pour les trois tâches : données, types numériques, formats,
classes explicites, index, prédictions optionnelles, limites et immutabilité.
Ils bloquent fit/predict/predict_proba, transformations, métriques et Optuna,
même après suppression du dataset original, dans les trois modes preprocessing.
L'import sans openpyxl et l'erreur d'installation sont également testés.
Les tests ranking vérifient les
calculs contrôlés de capture/lift et d'erreur régression, les scores identiques,
les effectifs non divisibles, les paramètres invalides, les copies et les index.
Ils couvrent les six modèles existants et les trois modes preprocessing.
Les prédictions initiales sont partagées avec le scoring sans nouvel appel au
modèle. Les diagnostics restent utilisables quand tout appel au modèle est
bloqué. Modifier les features ou cibles du holdout, avec sa frontière fixée,
change les diagnostics mais conserve CV, Optuna, baseline et gagnant.
Les tests de fuite des trois tâches tracent les instances, les indices vus
par fit/transform et les médianes propres à chaque fold. Ils vérifient le
fit final sur le train seulement et la stabilité de la sélection lorsque
seules les features du holdout changent.

La mémoire du OneHot dense basic et la matrice CI des versions minimales
(dont MLflow 3.1, non exécuté dans cet environnement)
restent à surveiller. Les colonnes entièrement manquantes dans un fold basic
restent une erreur ; les constantes basic peuvent avertir par fold/essai.
Openpyxl et MLflow restent dans leurs extras respectifs, sans dépendance obligatoire
supplémentaire. Aucun feature engineering automatique n'est introduit.
Datetime automatique, sélection de
features, outliers, logs, target encoding, encodage haute cardinalité, SHAP,
optimisation de seuil, calibration, CatBoost, Model Registry/deployment
et group/time split restent hors V0.7. Aucune fonctionnalité V0.8 n'est ajoutée.
