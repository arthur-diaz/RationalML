"""Install the built core wheel into a fresh venv and import outside the checkout."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import venv

from check_release import PUBLIC_API, ROOT, artifacts, audit_wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    args = parser.parse_args()
    wheel, _ = artifacts(args.directory.resolve(strict=True))
    wheelhouse = args.wheelhouse.resolve(strict=True)
    expected_version = audit_wheel(wheel)[0]["Version"]
    with TemporaryDirectory(prefix="rationalml-wheel-") as directory:
        root = Path(directory)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "--disable-pip-version-check", "install",
                        "--no-index", "--find-links", str(wheelhouse), str(wheel)], cwd=root, check=True)
        smoke = """
from importlib import metadata, util
from pathlib import Path
import sys
import pandas as pd
from rationalml import AutoML, AutoMLResult, MLflowConfig
import rationalml
assert rationalml.__version__ == metadata.version('RationalML') == sys.argv[1]
assert set(rationalml.__all__) == set(sys.argv[2].split(','))
for name in ('normalize_task', 'FeatureSchema', 'infer_schema', 'build_preprocessor'):
    assert not hasattr(rationalml, name), name
from rationalml.tasks import normalize_task
from rationalml.preprocessing import FeatureSchema, infer_schema, build_preprocessor
assert normalize_task('binary') is rationalml.TaskType.BINARY
schema = infer_schema(pd.DataFrame({'feature': [1., 2.]}))
assert isinstance(schema, FeatureSchema)
assert callable(build_preprocessor)
installed = metadata.metadata('RationalML')
assert installed.get_all('License-Expression') == ['Apache-2.0']
assert installed.get_all('License-File') == ['LICENSE']
assert Path(rationalml.__file__).is_relative_to(Path(sys.prefix))
for name in ('mlflow', 'shap', 'openpyxl', 'lightgbm', 'xgboost'):
    assert util.find_spec(name) is None, name
    assert name not in sys.modules, name
print('Clean core wheel:', rationalml.__version__, rationalml.__file__)
"""
        # -I excludes cwd, PYTHONPATH and user site packages even if the caller sets them.
        subprocess.run([str(python), "-I", "-c", smoke, expected_version, ",".join(sorted(PUBLIC_API))], cwd=root, check=True)
        example = root / "quickstart.py"
        shutil.copyfile(ROOT / "examples/quickstart.py", example)
        subprocess.run([str(python), "-I", str(example)], cwd=root, check=True)
        subprocess.run([str(python), "-m", "pip", "check"], cwd=root, check=True)


if __name__ == "__main__":
    main()
