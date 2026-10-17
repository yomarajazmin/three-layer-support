"""Every third-party package the project imports must be listed in requirements.txt."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Project-local top-level modules and packages, discovered so new ones never look third-party.
LOCAL = ({p.stem for p in ROOT.glob("*.py")} | {p.stem for p in (ROOT / "tests").glob("*.py")}
         | {p.name for p in ROOT.iterdir() if p.is_dir() and any(p.glob("*.py"))})
DIST_NAME = {"dotenv": "python-dotenv", "yaml": "pyyaml"}   # import name -> PyPI name


def project_imports() -> set[str]:
    names = set()
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names - LOCAL - set(sys.stdlib_module_names) - {"__future__"}


def required() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {re.split(r"[<>=!~\s\[#]", ln.strip(), maxsplit=1)[0].lower()
            for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def test_all_third_party_imports_are_in_requirements():
    missing = {DIST_NAME.get(n, n).lower() for n in project_imports()} - required()
    assert not missing, f"add to requirements.txt: {sorted(missing)}"
