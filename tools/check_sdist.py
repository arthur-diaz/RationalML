"""Rebuild a wheel from an isolated extracted sdist, with no network or checkout files."""

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from check_release import artifacts, audit_sdist, audit_wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheel, sdist = artifacts(args.directory.resolve(strict=True))
    version = audit_wheel(wheel)[0]["Version"]
    audit_sdist(sdist, version)
    with TemporaryDirectory(prefix="rationalml-sdist-") as directory:
        root = Path(directory)
        with tarfile.open(sdist) as archive:
            archive.extractall(root, filter="data")
        source = root / f"rationalml-{version}"
        output = root / "rebuilt"
        # Build dependencies are installed during environment setup, not here.
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                        "--outdir", str(output), str(source)], cwd=root, check=True)
        rebuilt, = output.glob("*.whl")
        original_metadata, original_files = audit_wheel(wheel)
        rebuilt_metadata, rebuilt_files = audit_wheel(rebuilt)
        if (original_files != rebuilt_files or original_metadata.items() != rebuilt_metadata.items()
                or original_metadata.get_payload() != rebuilt_metadata.get_payload()):
            raise ValueError("Rebuilt wheel content/metadata differs from the release wheel.")
        with ZipFile(wheel) as original, ZipFile(rebuilt) as replacement:
            for name in original_files:
                if name.endswith(".py") and original.read(name) != replacement.read(name):
                    raise ValueError(f"Rebuilt source differs: {name}")
        print("Isolated offline sdist rebuild verified:", rebuilt.name)


if __name__ == "__main__":
    main()
