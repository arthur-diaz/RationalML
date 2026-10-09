# Préparer une release RationalML

Cette checklist prépare des artefacts ; elle ne publie rien à elle seule.

**License: Apache-2.0.** RationalML is distributed under the Apache License 2.0.
LICENSE contient le texte officiel intégral téléchargé depuis Apache, sans
modification. pyproject.toml déclare license="Apache-2.0", license-files=["LICENSE"]
et setuptools>=77.0.3, sans ancien classifier de licence. Le contrôle de release
exige exactement ces métadonnées et conserve le texte à l'identique dans les
archives. Le fichier utilisateur LICENCE.txt est préservé hors des distributions.

Le dépôt officiel est https://github.com/arthur-diaz/RationalML. Il est privé,
confirmé par le mainteneur ; les liens Homepage/Source sont conservés et leur
accès public doit être vérifié avant publication. Aucun lien Issues/Changelog
non vérifié n'est ajouté aux métadonnées. Le nom RationalML reste **à vérifier
sur PyPI** : l'API publique n'a retourné aucun projet lors de l'audit, ce qui
ne garantit pas sa disponibilité. Aucun nom n'a été réservé.

## Checklist locale

- [x] Licence Apache-2.0 documentée, texte officiel dans LICENSE et métadonnées SPDX.
- [ ] Vérifier l'accès aux liens officiels et le nom PyPI avant première release.
- [ ] Working tree propre, changements relus et commités ; aucune donnée privée.
- [ ] Version dans pyproject.toml et entrée CHANGELOG concordantes.
- [ ] CI verte : core Python 3.10/3.11/3.12, extras complets et minima optionnels.
- [ ] Suite pytest complète et pip check, sans filtrage global des warnings.
- [ ] Build neuf, audit exact des archives/métadonnées et reconstruction sdist.
- [ ] Smoke test wheel dans un venv neuf, hors checkout, sans extras ni réseau.

Depuis un environnement de développement avec les extras nécessaires :

```sh
python -m pip install -e ".[test,boosting,excel,mlflow,shap]" build "setuptools>=77.0.3"
python -m pytest -q
python -m pip check
python examples/quickstart.py
python -m build --outdir dist/release
# Préparation réseau séparée ; les validations suivantes restent hors ligne.
python -m pip download --only-binary=:all: --dest wheelhouse dist/release/*.whl
python tools/check_release.py dist/release
python tools/check_sdist.py dist/release
python tools/check_wheel.py dist/release --wheelhouse wheelhouse
```

dist/release doit être neuf et contenir exactement un wheel et un sdist.
check_sdist utilise les outils de build déjà installés avec --no-isolation ;
check_wheel installe avec --no-index, utilise python -I hors dépôt, vérifie
les 11 exports, les quatre helpers dans leurs sous-modules, la licence Apache-2.0
et l'absence des cinq dépendances optionnelles, puis entraîne
réellement le quickstart core (CV 2, un trial), prédit et calcule le ranking.
Aucun script de validation ne télécharge de dataset ou n'appelle un service.

## Actions du mainteneur sur GitHub/PyPI

- [ ] Configurer l'environnement GitHub nommé **pypi**, avec ses protections.
- [ ] Dans PyPI, créer le Trusted Publisher (ou pending publisher pour un
  premier projet) : owner **arthur-diaz**, repository **RationalML**, workflow
  **release.yml**, environment **pypi**. Vérifier exactement ces quatre valeurs.
- [ ] Contrôler la concordance du tag envisagé avec v + version du wheel :
  `python tools/check_release.py dist/release --tag v<VERSION> --require-license`.
- [ ] Après toutes les validations, créer/pousser volontairement ce tag ; une
  GitHub Release et ses notes peuvent ensuite être ajoutées par le mainteneur.
- [ ] Vérifier le workflow de publication et la version réellement sur PyPI.
- [ ] Depuis un nouveau venv : installer la version publiée, vérifier
  importlib.metadata.version("RationalML"), import, quickstart et pip check.

Le workflow release.yml ne démarre que sur push d'un tag v*. Le job build
contrôle tag/version, licence, tests et distributions ; il transfère seulement
les deux fichiers validés. Le job publish télécharge le même artefact, sans
checkout ni reconstruction. Lui seul reçoit id-token: write et l'environnement
pypi ; pypa/gh-action-pypi-publish@release/v1 utilise OIDC et ses attestations
par défaut. Aucun PYPI_TOKEN, username/password ni secret permanent.

La configuration des comptes, environnements et protections distants **n'est
pas faite par ce changement**. Documentation officielle :
[Trusted Publishing PyPI](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
et [action de publication PyPA](https://github.com/pypa/gh-action-pypi-publish).

## État de cette préparation

Aucune publication PyPI/TestPyPI, aucun tag et aucun push ne sont effectués.
La licence Apache-2.0 est choisie ; l'accès public du dépôt, le nom PyPI et la
configuration Trusted Publisher restent des vérifications du mainteneur.
Les validations locales ne prétendent pas remplacer une exécution CI distante.

## Validation locale V1 (Windows)

| Environnement | Passed | Warnings | Skipped |
| --- | ---: | ---: | ---: |
| Full extras, Python 3.12.6 | 1250 | 2 | 0 |
| Core sans extras, Python 3.10.19 | 847 | 10 | 90 |
| SHAP 0.49.1 / XGBoost 3.0.5, Python 3.10.19 | 211 | 7 | 0 |
| MLflow 3.1.4 / SQLAlchemy 2.0.54, Python 3.12.6 | 110 | 28 | 0 |
| Openpyxl 3.0.10, Python 3.12.6 | 128 | 0 | 0 |

Les 1133 cas V0.10 sont conservés ; 117 cas sont ajoutés dans 25 fonctions.
La finalisation Apache-2.0 / 11 exports conserve les 1225 cas de la préparation
V1 et ajoute 25 cas. Les tests d'import des helpers avancés utilisent leurs
sous-modules ; les fixtures valides embarquent le vrai texte Apache-2.0.
Les deux tests historiques modifiés comparent désormais les versions avec
la source unique plutôt qu'une valeur 0.10.0 littérale. Les skips core sont
les importorskip des extras absents (dont 18 nouveaux cas de schéma boosting).
Les warnings concernent la classe positive implicite, les transformers de test
pandas 2 qui écrivent sur leur entrée et les dépréciations internes MLflow.
Aucun filtre global ne masque les warnings.

La suite complète termine en 324,70 secondes. pip check est propre dans les
quatre environnements et dans le venv wheel neuf. Quickstart, import isolé
python -I, mini-entraînement, prédiction, ranking et calibration passent sans
extras dans ce dernier ; installation --no-index et absence de dépendance au
checkout pour les imports. La reconstruction du sdist hors dépôt passe avec
--no-isolation, sans réseau : noms de fichiers, métadonnées et sources Python
du wheel reconstruit concordent. Python 3.11 est configuré en CI mais n'a pas
été exécuté localement ; aucun job distant n'est présenté comme exécuté.

Seul rationalml/__init__.py change dans le code de production : version issue
des métadonnées et liste racine limitée à 11 exports. Les quatre helpers restent
dans leurs sous-modules ; le moteur ML, ses signatures et ses transformations
ne changent pas.

## Audit des distributions

Le wheel contient exactement **46 fichiers** : 35 modules RationalML,
6 modules legacy intentionnels et 5 fichiers dist-info, dont LICENSE. Aucun test, workflow,
cache, dataset, notebook, fichier build ou temporaire. Les six modules legacy
restent nécessaires à la compatibilité annoncée et testée ; ils ne font pas
partie des 11 exports stables. sub_area/sub_area_pr et les helpers dangereux
continuent de lever LegacyAPIError.

```text
metrics.py
report.py
sco_mod.py
scoring.py
second_step.py
strategy.py
rationalml/__init__.py
rationalml/automl.py
rationalml/config.py
rationalml/data.py
rationalml/exceptions.py
rationalml/legacy.py
rationalml/result.py
rationalml/evaluation/__init__.py
rationalml/evaluation/baseline.py
rationalml/evaluation/calibration.py
rationalml/evaluation/metrics.py
rationalml/evaluation/predictions.py
rationalml/evaluation/ranking.py
rationalml/evaluation/registry.py
rationalml/explainability/__init__.py
rationalml/explainability/shap.py
rationalml/models/__init__.py
rationalml/models/base.py
rationalml/models/registry.py
rationalml/optimization/__init__.py
rationalml/optimization/cv.py
rationalml/optimization/optimizer.py
rationalml/optimization/spaces.py
rationalml/preprocessing/__init__.py
rationalml/preprocessing/builder.py
rationalml/preprocessing/config.py
rationalml/preprocessing/schema.py
rationalml/reporting/__init__.py
rationalml/reporting/config.py
rationalml/reporting/excel.py
rationalml/tasks/__init__.py
rationalml/tasks/base.py
rationalml/tracking/__init__.py
rationalml/tracking/config.py
rationalml/tracking/mlflow.py
rationalml-1.0.0.dist-info/METADATA
rationalml-1.0.0.dist-info/RECORD
rationalml-1.0.0.dist-info/WHEEL
rationalml-1.0.0.dist-info/top_level.txt
rationalml-1.0.0.dist-info/licenses/LICENSE
```

Le sdist contient **106 fichiers** : ces sources, les 48 tests/helpers Python
(notamment conftest.py et _robustness.py), pyproject.toml, MANIFEST.in, README.md,
CHANGELOG.md, RELEASE.md, LICENSE, examples/quickstart.py, les trois outils de validation,
PKG-INFO, setup.cfg et les cinq fichiers egg-info générés. .github est exclu
explicitement pour neutraliser aussi un ancien cache SOURCES.txt. Aucun fichier
de licence autre que LICENSE n'est inclus ; son texte reste identique à celui du projet.

Métadonnées réellement inspectées :

| Champ | Valeur |
| --- | --- |
| Name / Version | RationalML / 1.0.0 |
| Requires-Python | >=3.10 |
| Description-Content-Type | text/markdown |
| Author | Arthur D., identité des commits ; aucun email inventé |
| Provides-Extra | boosting, legacy, excel, mlflow, shap, test |
| Project-URL | Homepage et Source : URL officielle du dépôt privé |
| License-Expression / License-File | Apache-2.0 / LICENSE, exactement une occurrence de chaque |
| Classifiers | Python 3, 3.10/3.11/3.12, OS Independent, Scientific/Engineering AI |

Les cinq Requires-Dist core restent numpy>=1.24,<3, pandas>=2,<4,
scikit-learn>=1.5,<2, optuna>=4,<6 et threadpoolctl>=3.5,<4. Tous sont utilisés
(calculs, tableaux, modèles/CV, optimisation et contrôle des threads).
Les autres Requires-Dist sont conditionnés par leurs extras : lightgbm>=4,<5
et xgboost>=2,<4 ; tqdm>=4 ; openpyxl>=3.0.10,<4 ; mlflow>=3.1,<4 ;
shap>=0.49.1,<0.50 ; pytest>=8,<10. Aucune borne runtime n'est modifiée.

Compromis déjà connus : SHAP 0.49.x exige XGBoost <3.1 pour les intercepts
vectoriels ; MLflow 3.1 exige SQLAlchemy <2.1 dans sa validation minimum.
Ces restrictions sont documentées ; fit/predict XGBoost moderne restent valides.
Le contrôle --require-license exige explicitement Apache-2.0 avec LICENSE ;
l'absence, une autre expression, des doublons ou un fichier inattendu sont refusés.

Les artefacts de cette finalisation sont dans dist/v1-final : exactement un
wheel et un sdist. Le build isolé a utilisé setuptools 84.0.0, satisfaisant
la borne setuptools>=77.0.3. Le chemin de licence dist-info indiqué plus haut
a été vérifié dans le wheel réellement généré. Les textes LICENSE du wheel
et du sdist sont comparés octet par octet au fichier du projet ; SHA-256 :
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.

Le doublon de section [project] initialement présent a été fusionné pour rendre
le TOML valide, en conservant les paramètres Apache-2.0 choisis. Aucune logique
ML ne change. Aucun tag, publication, push ou paramétrage de compte distant
n'est effectué ; l'accès public du dépôt, le nom PyPI et la configuration du
Trusted Publisher/environnement pypi restent des vérifications du mainteneur.
