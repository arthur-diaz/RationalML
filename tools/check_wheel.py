"""Install the built core wheel into a fresh venv and import outside the checkout."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import venv


def main():
    wheel = Path(sys.argv[1]).resolve(strict=True)
    if wheel.suffix != ".whl":
        raise ValueError("Pass the built wheel path.")
    with TemporaryDirectory(prefix="rationalml-wheel-") as directory:
        root = Path(directory)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "--disable-pip-version-check", "install", str(wheel)], cwd=root, check=True)
        smoke = """
from importlib import metadata, util
from pathlib import Path
import sys
from rationalml import AutoML, AutoMLResult, MLflowConfig
import rationalml
assert rationalml.__version__ == metadata.version('RationalML') == '0.10.0'
assert Path(rationalml.__file__).is_relative_to(Path(sys.prefix))
for name in ('mlflow', 'shap', 'openpyxl', 'lightgbm', 'xgboost'):
    assert util.find_spec(name) is None, name
    assert name not in sys.modules, name
print('Clean core wheel:', rationalml.__version__, rationalml.__file__)
"""
        # -I excludes cwd, PYTHONPATH and user site packages even if the caller sets them.
        subprocess.run([str(python), "-I", "-c", smoke], cwd=root, check=True)
        subprocess.run([str(python), "-m", "pip", "check"], cwd=root, check=True)


if __name__ == "__main__":
    main()
