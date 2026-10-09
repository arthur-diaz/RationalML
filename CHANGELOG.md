# Changelog

L'historique ci-dessous repose sur les tags, commits et contrats présents dans
ce dépôt ; il ne présume pas de publications PyPI. Les correctifs 0.1.x et
le recentrage 0.2.1 n'ont pas de tag distinct dans l'historique disponible.

## Unreleased — préparation 1.1

- Contrat CPU existant vérifié : n_jobs 1/N/-1, trials et folds séquentiels,
  paramètres des arbres et limites BLAS/OpenMP temporaires. Les méthodes
  result.predict* appliquent désormais aussi ces limites natives.
- verbose compatible V1 : 0 pour le silence, 1 pour progression/résumé, >=2 pour les détails
  Optuna/résumé. Restauration des niveaux de logging même en cas d'exception,
  sans modification des handlers utilisateur ni des filtres de warnings.
- Progression en bibliothèque standard, fondée sur les trials terminés,
  y compris échecs/pruning ; budgets incomplets et interruptions explicites.
- AutoMLResult.fit_time, model_fit_times et summary() -> str, à partir des
  résultats stockés ; aucun changement des schémas de tableaux ni des 11 exports.
- Espace LightGBM moderne corrigé après audit : régularisation/gain réduits,
  feuilles cohérentes avec la profondeur, taux d'apprentissage logarithmique
  et sous-échantillonnages moins agressifs. Domaines legacy conservés.
- min_child_samples moderne : [5, 100] → [5, 50], avec suggest_int linéaire.
  Inclut le défaut LightGBM 20, la flexibilité de 5 et la régularisation de 50,
  pour limiter les trials peu exploitables sur petits datasets, indépendamment
  de la tâche et de Credit Card Fraud. Aucun autre domaine LightGBM modifié
  lors de cet ajustement ; domaines legacy conservés à [5, 100].
- Version pyproject.toml maintenue à 1.0.0 ; aucun tag ni publication 1.1.

## 1.0.0

RationalML 1.0 stabilise le contrat public. Sa philosophie reste : données
préparées par le Data Scientist → comparaison robuste des modèles → diagnostics
et reporting post-fit. Le feature engineering reste sous contrôle utilisateur.

- Stabilisation des exports, signatures, attributs documentés et schémas de
  tableaux existants ; aucune fonctionnalité ML ajoutée.
- Version unique dans pyproject.toml, exposée via les métadonnées installées.
- Quickstart core autonome, politique SemVer, documentation de release et
  historique vérifiable.
- Audits exacts des distributions, reconstruction sdist hors dépôt et smoke
  test wheel dans un venv neuf sans dépendances optionnelles, hors ligne.
- Workflow tag-only de Trusted Publishing avec artefacts validés réutilisés,
  contrôle tag/version et validation exacte de la licence Apache-2.0 / LICENSE.
- Apache-2.0 license selected for the first stable release ; texte officiel
  intégral dans LICENSE, métadonnées SPDX et backend setuptools>=77.0.3.
- API racine limitée à 11 exports ; normalize_task, FeatureSchema, infer_schema
  et build_preprocessor restent disponibles dans leurs sous-modules avancés.
- Maintien intentionnel des six modules legacy, hors contrat stable V1.
- Tous les tests V0.10 restent présents ; les deux attentes de version
  littérales sont remplacées par une comparaison avec la source unique.
- Les tests d'import utilisent les sous-modules pour les quatre helpers retirés
  de la racine ; les audits/test fixtures exigent Apache-2.0 et LICENSE.

Aucune publication, création de tag ou configuration de compte distant n'est
effectuée ici.

## 0.10.0

- Durcissement de l'isolation train/test, des folds partagés et du choix CV.
- Tests de reproductibilité, d'absence de mutation et des signatures publiques.
- Erreurs explicites pour configurations/données invalides ; exceptions SHAP
  conservées ; remplacement atomique Excel éprouvé en cas d'échec.
- CI core, extras complets, minima optionnels et packaging.

## 0.9.0

- SHAP post-fit optionnel avec X et background explicites, sortie unique et
  class_label obligatoire en multiclass ; aucun refit ni état explicatif stocké.
- Restriction documentée SHAP 0.49.x / intercepts vectoriels XGBoost >=3.1.

## 0.8.0

- Diagnostics agrégés de calibration sur probabilités du holdout conservées :
  Brier, ECE, MCE et tables par bins, sans recalibrer les modèles.
- Intégration optionnelle de ces diagnostics aux rapports Excel et MLflow.

## 0.7.0

- Tracking MLflow optionnel, runs parent/enfants et résultats déjà calculés.
- model_best_params par candidat ; prédictions individuelles sur opt-in.

## 0.6.0

- Export Excel optionnel avec configuration dédiée et feuilles numériques.
- Les anciens helpers Excel dangereux restent explicitement désactivés.

## 0.5.0

- Prédictions du holdout conservées avec index original et accès par copie.
- Ranking et top_segment, classification one-vs-rest explicite et régression.

## 0.4.0

- Baseline naïve calculée sur les folds train partagés, hors sélection.
- Diagnostics de stabilité CV et amélioration par rapport à la baseline.

## 0.3.0

- Support multiclass et régression, modèles/métriques filtrés par tâche.
- task auto conservateur et métriques automatiques spécifiques à la tâche.
- Probabilités multiclass dans l'ordre de classes_, importance logistique longue.

## 0.2.1

- Recentrage documenté sur des données préparées par le Data Scientist.
- preprocessing=None par défaut, basic de convenance, transformer utilisateur
  cloné par fold ; retrait de la valeur publique preprocessing="auto".
- Ce contrat figure dans le code et les notes de migration V0.3 ; aucun tag
  0.2.1 autonome n'a été retrouvé.

## 0.2.0

- Preprocessing tabulaire optionnel avec schéma, imputation, OneHotEncoding
  et scaling conditionnel ; pipelines fitted dans chaque fold CV.
- Inférence sur colonnes brutes et importance avec noms transformés/source.

## 0.1.2

Aucun tag, commit de release ou métadonnée distincte 0.1.2 n'a été retrouvé.
Aucun changement n'est attribué artificiellement à cette version.

## 0.1.1

- Classe positive explicite, encodage interne 1 et predict_positive_proba.
- Désactivation de sub_area/sub_area_pr par LegacyAPIError : contrat ambigu
  et anciens calculs incorrects.
- Ces changements sont présents dans le code portant la version 0.1.1 au tag
  historique v0.1.0 ; le tag et la version source ne coïncident donc pas.

## 0.1.0

- Fondations AutoML/Config/Result, registries modèles/métriques et Optuna.
- Split holdout avant optimisation, CV exclusivement sur train et gagnant CV.
- Classification binaire, LogisticRegression et boosting optionnel.
- Le tag v0.1.0 couvre également les corrections 0.1.1 décrites ci-dessus.
