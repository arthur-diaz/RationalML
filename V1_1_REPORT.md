# Préparation RationalML 1.1 — runtime et terminal

## 1. Branche et périmètre

Branche : `feat/v1.1-runtime-ux`, déjà présente lors de l'inspection initiale.
Worktree initial propre. Après `git fetch origin main`, HEAD, main et origin/main
pointaient tous sur `63e9a026cad2608599c809c4625fcf40bc7071db`.
Aucun merge, tag, push, publication PyPI ni bump de version. pyproject.toml
reste à 1.0.0, avec les mêmes dépendances et la même licence Apache-2.0.

Le split, les folds, baseline, TPE seedé, NopPruner, sélection exclusivement
CV, refit sur TRAIN et évaluation unique TEST conservent leur protocole V1.
Les diagnostics post-fit restent fondés sur les prédictions stockées.

## 2. Fichiers

```text
ml/
├── README.md                              modifié
├── CHANGELOG.md                           modifié : Unreleased
├── V1_1_REPORT.md                          nouveau : ce rapport
├── rationalml/
│   ├── automl.py                          modifié : affichage et chronométrage
│   ├── result.py                          modifié : durées, summary, CPU en inférence
│   ├── runtime.py                         nouveau : terminal et logging temporaire
│   └── optimization/
│       ├── optimizer.py                   modifié : logging et callback de progression
│       └── spaces.py                      modifié : domaine LightGBM moderne
└── tests/
    ├── test_runtime_resources.py           nouveau
    ├── test_runtime_output.py              nouveau
    └── test_lightgbm_search_space.py        nouveau
```

Aucun test V1.0 n'a été retiré ni modifié. Aucun changement dans les fichiers
models, metrics, preprocessing, exports racine, workflows CI/release ou licence.
Les artefacts et données d'acceptation restent sous `.venv/validation` et
`dist/runtime-ux`, ignorés par Git.

## 3. n_jobs : avant / après

Avant : validation 1/N/-1 déjà correcte ; refus des booléens, 0 et <-1.
ModelSpec transmettait déjà n_jobs aux classifieurs/régresseurs LightGBM/XGBoost.
LogisticRegression/lbfgs et Ridge avaient `n_jobs_parameter=None` et bénéficiaient
déjà de threadpoolctl pendant CV, refit et évaluation. Trials Optuna n_jobs=1
et folds séquentiels. L'inférence ultérieure result.predict* n'avait pas de
limite native temporaire pour les modèles linéaires.

Après : même protocole et même validation, désormais prouvés par les tests.
result.predict/predict_proba utilisent aussi des limites natives temporaires ;
predict_positive_proba délègue à predict_proba. Les limites sont restaurées
même sur erreur. Aucun n_jobs inventé pour Ridge ou ajouté à lbfgs.

1/N limite chaque pool BLAS/OpenMP contrôlé et les threads des arbres.
-1 est transmis aux arbres selon leurs conventions ; threadpoolctl(limits=None)
laisse les réglages natifs/externe en vigueur. Ce n'est pas une affinité CPU
globale ni une garantie sur les threads créés par un transformer personnalisé.
Config et paramètres de registry utilisateur restent préservés.

## 4–6. verbose, progression et durées

Avant : verbose était un entier >=0. Il pilotait un logger RationalML INFO
souvent invisible sans configuration ; les INFO Optuna apparaissaient même à 0.
Les arbres intégrés étaient déjà configurés pour rester silencieux.

Après : 0 = aucune sortie normale ; 1 = présentation + progression + résumé ;
>=2 = présentation + détails Optuna INFO + résumé, sans barre. La validation
V1.0 est conservée sans borne supérieure : config.py est identique à main.
Les niveaux 2, 3 et 10 sont testés ; négatifs et booléens restent refusés.
Les niveaux
Optuna et de ses enfants existants sont restaurés dans finally. Handlers,
filtres, formatters, propagation et disabled utilisateur ne sont pas modifiés.
Les warnings Python et erreurs importantes gardent leur comportement standard.
Aucune agrégation de warnings ajoutée, afin de préserver les filtres utilisateur.

La barre ASCII utilise uniquement print/sys et le callback public Study.optimize.
Total = modèles × n_trials ; chaque état terminal COMPLETE/FAIL/PRUNED compte
une fois. Le finally réconcilie les échecs non gérés pour lesquels Optuna ne
déclenche pas ses callbacks. Une erreur reste propagée. RUNNING n'est pas
compté. Timeout, arrêt anticipé et interruption conservent le compteur réel.
Terminal : réécriture de ligne ; sortie redirigée : une ligne par modèle,
puis fermeture. Aucun nouveau tqdm/rich/joblib ajouté aux dépendances.

API en secondes, mesurée par time.perf_counter :

- `result.fit_time: float` : entrée de fit jusqu'à la construction complète du
  résultat, incluant validation/copies, split, baseline, CV/Optuna, refit,
  TEST, importance et présentation/progression éventuelle. Exclut l'impression
  du résumé final et les opérations post-fit.
- `result.model_fit_times: dict[str, float]` : chaque optimisation complète,
  avec construction des pipelines, fits et scores CV. Le gagnant inclut aussi
  la construction et le fit final sur TRAIN. Exclut split, baseline, TEST et
  importance. Ordre des modèles demandés ; aucun refit des non-gagnants.

Aucune colonne ajoutée à leaderboard/cv_results ; durées disponibles à verbose=0.

## 7. summary

`result.summary() -> str` retourne le classement CV, gagnant, métrique primaire,
score TEST et durées stockées. Ne fait aucun print, fit, predict, ranking,
calibration ou export et ne modifie pas les résultats. Le fit verbose>=1
imprime cette même chaîne. L'utilisateur peut faire `print(result.summary())`.

## 8. Audit LightGBM avant modification

Audit basé sur le code réel de spaces.py et ModelRegistry, puis les
[paramètres officiels LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
Les domaines sont une décision conservatrice de RationalML, pas une
recommandation universelle ni une optimisation sur les datasets d'acceptation.

| Paramètre | Ancien domaine moderne | Nouveau domaine moderne | Analyse et justification |
|---|---|---|---|
| max_depth | entier 2–20 | entier 2–12 | 20 permet des arbres très profonds ; budget borné tout en conservant plusieurs capacités. |
| num_leaves | entier 16–256, uniforme et indépendant | entier 4–min(128, 2**max_depth), logarithmique | À profondeur 2, 16–256 ne peut pas être réalisé ; inclut désormais de petits arbres et évite les combinaisons structurellement redondantes. |
| learning_rate | 0.005–0.5, uniforme | 0.01–0.2, logarithmique | Avec 100 arbres fixes, les extrêmes très faibles apprennent peu et les taux élevés peuvent être instables ; davantage de masse sur les taux modérés. |
| reg_alpha | 1e-8–1000, logarithmique | 1e-8–10, logarithmique | Plusieurs centaines de L1 peuvent annuler les faibles gradients, notamment avec peu de positifs ; conserve une régularisation de presque nulle à substantielle. |
| reg_lambda | 1e-8–1000, logarithmique | 1e-8–10, logarithmique | Une L2 massive amortit fortement les updates ; plage logarithmique large conservée. |
| min_split_gain | 1e-8–100, uniforme | 0–0.1, uniforme | Tirage typique autour de 50 : interdit beaucoup de splits utiles. Gain absolu dépendant de l'objectif et de l'échelle, donc pas de garantie universelle d'absence de dégénérescence. |
| min_child_samples | entier 5–100, linéaire | entier 5–50, linéaire (`suggest_int`) | 20 est le défaut LightGBM ; 5 apporte de la flexibilité et 50 conserve une régularisation importante. Réduit les trials limitant fortement les splits sur petits datasets, indépendamment de la tâche (binary / multiclass / regression) et de Credit Card Fraud. |
| colsample_bytree | 0.1–0.7, uniforme | 0.5–1, uniforme | 10 % peut réduire très fortement les possibilités avec peu de variables ; inclut désormais toutes les variables. |
| subsample | 0.3–1, uniforme | 0.6–1, uniforme | Évite de retirer 70 % des lignes à chaque bagging, notamment lorsque les événements sont rares ; ne rééquilibre pas les classes. |
| subsample_freq | entier 0–15 | entier 1–5 | 0 désactivait subsample malgré son optimisation ; bagging actif lorsque subsample<1, avec fréquences modérées. subsample=1 garde toutes les lignes. |

Autres réglages conservés : boosting_type=gbdt, n_estimators=100,
verbosity=-1, deterministic=True, force_col_wise=True, random_state et n_jobs.
Les objectifs restent binary / multiclass / regression. Aucun early stopping,
class_weight, scale_pos_weight ou réglage GPU ajouté. XGBoost, LR et Ridge
conservent leurs domaines existants.

Constat reproductible AVANT modification avec le TPE seed=42 : les cinq
premiers min_split_gain étaient environ 60.11, 43.19, 59.24, 68.42, 54.67 ;
les deux premiers reg_alpha environ 286.98 et 466.67. Cette combinaison
justifiait la correction avant consultation des performances d'acceptation.

Les sept domaines legacy restent intacts. suggest_lightgbm(trial, bounds=...)
conserve les bornes et distributions explicitement fournies ; seule la valeur
par défaut du chemin moderne change. Les tests vérifient distributions,
bounds, diversité et construction/fit pour les trois tâches, sans score minimum.

Dernier ajustement demandé : seul min_child_samples du chemin moderne passe de
[5, 100] à [5, 50], avec suggest_int linéaire (pas de logarithme, pas de pas
supérieur à 1). Les tests de bornes contrôlent également IntDistribution et
step=1 ; le contrat legacy reste [5, 100]. Aucun autre domaine LightGBM modifié
lors de cet ajustement.

Périmètre de cette dernière passe : spaces.py, test_lightgbm_search_space.py,
README.md, CHANGELOG.md et ce rapport uniquement. Les autres modifications V1.1
déjà présentes dans le worktree sont conservées sans changement.

## 9–13. Validation

Référence exécutée AVANT tout changement de code : **1250 passed, 2 warnings,
0 skipped, 320.42 s**, Python 3.12.6, tous extras installés.
Ajout : **102 cas** dans trois fichiers, dont 24 cas nets supplémentaires lors
de la passe de compatibilité verbose. Total final : **1352**.

| Environnement / suite | Passed | Warnings | Skipped | Durée |
|---|---:|---:|---:|---:|
| Python 3.12.6, tous extras, suite complète après compatibilité verbose, avant borne [5, 50] | 1352 | 2 | 0 | 341.21 s |
| Python 3.12.6, search space ciblé après borne min_child_samples [5, 50] | 8 | 0 | 0 | 0.91 s |
| Python 3.12.6, tous extras, suite complète finale après borne min_child_samples [5, 50] | 1352 | 2 | 0 | 370.17 s |
| Python 3.10.19, core, suite complète après compatibilité verbose | 926 | 10 | 113 | 67.18 s |
| Python 3.10.19, SHAP 0.49.1 / XGBoost 3.0.5, SHAP + contrats compatibles verbose | 313 | 7 | 0 | 47.61 s |
| Python 3.12.6, MLflow 3.1.4 / SQLAlchemy 2.0, tests MLflow | 110 | 28 | 0 | 111.08 s |
| Python 3.12.6, openpyxl 3.0.10, tests Excel | 128 | 0 | 0 | 18.12 s |

Les skips core correspondent aux dépendances optionnelles absentes. La suite
core ne collecte pas les modules de tests dont importorskip intervient au
niveau du module ; ses nombres ne doivent donc pas être additionnés au total
de la suite complète. Les tests ciblés supplémentaires ne s'additionnent pas
non plus à la suite complète qui les contient déjà.

Passe de compatibilité verbose : validation V1.0 exactement restaurée dans
AutoMLConfig (entier >=0, aucune borne supérieure, booléens refusés). Le logger
Optuna utilise INFO pour tout verbose>=2. Les tests couvrent 0/1/2/3/10,
ainsi qu'un entier 10**30 en configuration. La comparaison des cinq niveaux
pour binary/multiclass/regression et sklearn/LightGBM/XGBoost vérifie mêmes
splits, paramètres, classement CV, métriques et prédictions, et aucune mutation
du DataFrame. Les restaurations de logging sur succès/erreur et les filtres
de warnings sont également vérifiés à 3 et 10.

Tests ciblés de cette passe : **149 passed** sous Python 3.12 tous extras ;
**108 passed, 8 skipped** sous Python 3.10 core. Les suites complètes et le
packaging ont été relancés et passent sur le code compatible avant tout commit.
Les chiffres des minima MLflow/Excel proviennent de la validation initiale
V1.1 ; leurs tests sont également présents dans la suite complète finale.

Une première exécution après changements avait révélé deux attentes existantes
sur les arguments exacts de Study.optimize. La correction conserve l'appel
historique sans callbacks lorsque la progression est inactive ; les tests
existants sont conservés tels quels. La suite complète finale passe sur cette
correction, ainsi que les tests de baseline/progression ciblés.

Commandes principales :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_lightgbm_search_space.py -q -p no:cacheprovider --tb=short
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short
.\.venv\validation\core310\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short
.\.venv\Scripts\python.exe -m build --outdir dist/runtime-ux --no-isolation
.\.venv\Scripts\python.exe tools/check_release.py dist/runtime-ux --require-license
.\.venv\Scripts\python.exe tools/check_sdist.py dist/runtime-ux
.\.venv\Scripts\python.exe tools/check_wheel.py dist/runtime-ux --wheelhouse .venv/validation/v1-final-wheelhouse
.\.venv\Scripts\python.exe -m pip check
```

pip check : aucun conflit dans .venv, core310, shap310, mlflow31 et le venv
neuf du wheel. Aucun changement de dépendance pour obtenir ce résultat.

Python 3.11 n'est pas installé localement ; aucun résultat local 3.11 annoncé.
La matrice CI existante 3.10/3.11/3.12 reste inchangée et n'a pas été exécutée
à distance dans cette session.

Packaging : wheel/sdist construits sans isolation réseau ; audit des fichiers,
licence et métadonnées ; reconstruction sdist hors dépôt ; installation wheel
hors ligne dans un venv neuf, exports racine, absence d'extras, quickstart réel
et pip check : tous réussis. **47 fichiers wheel, 110 fichiers sdist**.
Archives locales de préparation seulement, sans publication.

## 14. Smoke tests réels

Les smoke tests ci-dessous et l'expérience Credit Card Fraud de la section 15
ont été exécutés avant le dernier ajustement min_child_samples à [5, 50], avec
la borne moderne [5, 100]. Ils restent des observations historiques et n'ont
pas été relancés ni utilisés pour choisir cette borne. Les validations core,
minima optionnels et packaging précédentes datent également d'avant cet ajustement ;
seules les nouvelles exécutions search space et suite complète valident [5, 50].

Breast Cancer / Wine / Diabetes, basic, cv=3, n_trials=5, seed=42, n_jobs=1,
modèles linéaires + LightGBM + XGBoost : tous terminent. Predict et probabilités
fonctionnent pour leurs tâches ; ranking, top_segment et calibration vérifiés.

| Dataset | Gagnant CV | Score CV primaire | Score TEST primaire |
|---|---|---:|---:|
| Breast Cancer | logistic_regression | ROC-AUC 0.994763 | 0.996032 |
| Wine | logistic_regression | F1 macro 0.986295 | 1.000000 |
| Diabetes | ridge | RMSE 55.741525 | 53.664509 |

Wine, ancien espace LightGBM : cinq scores F1 macro exactement égaux au
baseline 0.190964. Nouveau : meilleur CV 0.959417. Observations descriptives,
sans assertion de performance automatique ni ajustement des domaines ensuite.

Le script utilisateur `test_ml/acceptance_test.py` passe intégralement avec
Excel pour les trois tâches, MLflow SQLite local, SHAP linéaire binaire,
multiclasse et Ridge, ainsi que les erreurs attendues de régression.
Log conservé sous `.venv/validation/runtime-acceptance-output.txt`.

## 15. Credit Card Fraud

Source CSV référencée par la
[documentation officielle TensorFlow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
Seul le fichier est utilisé, sans installation de TensorFlow ni reprise des
transformations, rééquilibrages ou choix de seuil de ce tutoriel.
Les téléchargements OpenML incomplets ont été détectés et exclus.

CSV validé : 284807 lignes, 31 colonnes, 492 fraudes, 150828752 octets.
SHA256 : `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
Préparation explicite du DataFrame par le script d'acceptation :
drop_duplicates + reset_index, donnant 283726 lignes et 473 fraudes.
Aucune suppression ou transformation supplémentaire dans le moteur.

Options : binary, positive_class=1, average_precision, basic, cv=3, n_trials=5,
seed=42, n_jobs=1, modèles LR/LightGBM/XGBoost. Les 15 trials terminent.
Ancien espace évalué ensuite uniquement sur le même TRAIN et les mêmes folds ;
aucun appel holdout dans cette comparaison historique et aucun domaine retouché.

| Trial LightGBM | Avant, AP CV | Après, AP CV |
|---|---:|---:|
| 0 | 0.001665 | 0.049686 |
| 1 | 0.001665 | 0.307434 |
| 2 | 0.152899 | 0.809961 |
| 3 | 0.489997 | 0.773049 |
| 4 | 0.001854 | 0.831231 |

Classement exclusivement CV : XGBoost 0.845033, LightGBM 0.831231,
LogisticRegression 0.753018 ; baseline 0.001665. Les deux premiers trials
LightGBM restent faibles : pas de garantie qu'un tirage apprenne suffisamment,
notamment avec une cible très rare et un petit budget.

TEST du seul gagnant XGBoost : ROC-AUC 0.973544, AP 0.811635,
precision 0.958904, recall 0.736842, F1 0.833333. Ranking et top_segment
fonctionnent ; top 1 % capture 84.21 % des fraudes holdout.
Fit total 140.24 s ; LR 17.23 s, LightGBM 45.04 s, XGBoost 77.14 s
(ce dernier inclut le refit final). Ces durées sont descriptives de cette machine.
Résultats détaillés : `.venv/validation/fraud_runtime_results.json` ; script
rejouable `.venv/validation/run_fraud_runtime.py`. Données non distribuées.

## 16–17. Warnings, compatibilité et limites

Warnings existants conservés : classe positive implicite ; dépréciation MLflow /
SQLAlchemy ; SettingWithCopyWarning dans les transformers de tests sur pandas 2.
Le script d'acceptation conserve aussi l'avertissement natif MLflow sur pickle.
SHAP 0.49.1 avec XGBoost moderne et SQLAlchemy 2.1 avec MLflow 3.1 gardent
leurs limitations de compatibilité déjà documentées ; aucun changement des bornes.

API additive : fit_time, model_fit_times, summary(). Onze exports racine
strictement conservés ; aucun helper avancé réintroduit à la racine.
verbose conserve sa compatibilité V1.0, sans borne supérieure. L'espace moderne
LightGBM et donc ses paramètres/prédictions peuvent changer à seed identique
entre V1.0 et cette préparation. Reproductibilité intra-version conservée.
Les durées varient ; les budgets timeout peuvent modifier le nombre de trials
selon la vitesse de la machine et le coût d'affichage.

Logging Optuna et threadpoolctl utilisent des réglages au niveau processus :
pas de garantie d'isolation vis-à-vis de fits externes simultanés dans les
mêmes threads/processus. Trials/folds internes restent séquentiels.
Pas d'agrégation de warnings, nouveaux diagnostics ML, feature engineering,
rééquilibrage, sélection de features, threshold optimization ou GPU.

## 18–20. État Git final

`git diff --stat` (fichiers déjà suivis) :

```text
 CHANGELOG.md                         |  22 +++++++
 README.md                            | 107 ++++++++++++++++++++++++++++++++++-
 rationalml/automl.py                 |  39 +++++++++----
 rationalml/optimization/optimizer.py |  28 ++++++---
 rationalml/optimization/spaces.py    |  30 +++++++++-
 rationalml/result.py                 |  24 +++++++-
 6 files changed, 228 insertions(+), 22 deletions(-)
```

Nouveaux fichiers non encore suivis, en complément du diff : runtime.py
86 lignes, test_runtime_resources.py 112, test_runtime_output.py 283,
test_lightgbm_search_space.py 69, et ce rapport.

`git status --short` :

```text
 M CHANGELOG.md
 M README.md
 M rationalml/automl.py
 M rationalml/optimization/optimizer.py
 M rationalml/optimization/spaces.py
 M rationalml/result.py
?? V1_1_REPORT.md
?? rationalml/runtime.py
?? tests/test_lightgbm_search_space.py
?? tests/test_runtime_output.py
?? tests/test_runtime_resources.py
```

git diff --check ne signale aucune erreur. Aucun commit effectué automatiquement :
modifications laissées pour revue. Dernier commit existant inchangé :
`63e9a02 fix: quote pip download command in release workflow`.
