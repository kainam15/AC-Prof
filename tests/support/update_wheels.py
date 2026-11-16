"""Tiny executable distributions for real uv operations without network or models."""
from __future__ import annotations

import zipfile
from pathlib import Path


def write_wheel(index: Path, version: str, *, command_version: str | None = None) -> Path:
    project = index / "acprof"
    project.mkdir(parents=True, exist_ok=True)
    dist = f"acprof-{version}.dist-info"
    files = {
        "acprof/__init__.py": f'__version__ = "{version}"\n',
        "acprof/__main__.py": (
            f'def main():\n    print("AC-Prof {command_version or version}")\n'
            'if __name__ == "__main__":\n    main()\n'
        ),
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: acprof\nVersion: {version}\nRequires-Python: >=3.12\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist}/entry_points.txt": "[console_scripts]\nacprof=acprof.__main__:main\n",
    }
    files[f"{dist}/RECORD"] = "".join(f"{name},,\n" for name in files)
    path = project / f"acprof-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    (project / "index.html").write_text("\n".join(
        f'<a href="{wheel.name}">{wheel.name}</a>' for wheel in sorted(project.glob("*.whl"))
    ))
    return path
