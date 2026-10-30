"""Architecture boundary: the report layer cannot reach into the safety core.

P5-T07. The safety claim is not "the LLM is usually downstream" — it is that the
import graph makes an in-loop call impossible. ``fusion/`` and ``orchestrator/``
are scanned at the AST level, including dynamic ``importlib`` calls, so renaming
a module or hiding an import in a string does not slip past.

``pipeline.py`` is deliberately not scanned: it is the composition root that
writes the incident *after* the actions are assembled.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "cryoguard"
SAFETY_PACKAGES = ("fusion", "orchestrator")
FORBIDDEN_PREFIXES = ("cryoguard.report", "openai")


def python_files(package: str) -> list[Path]:
    files = sorted((SRC / package).rglob("*.py"))
    assert files, f"no python files found under {package}; the scan would pass vacuously"
    return files


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else ""
            )
            if name in {"import_module", "__import__"}:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        modules.add(argument.value)

    return modules


@pytest.mark.parametrize("package", SAFETY_PACKAGES)
def test_safety_core_never_imports_the_report_or_an_llm_client(package: str) -> None:
    violations: list[str] = []
    for path in python_files(package):
        for module in imported_modules(path):
            if module.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == [], (
        "the safety core must not reach the reporting layer or a model client:\n"
        + "\n".join(violations)
    )


def test_fusion_is_a_pure_decision_layer_with_no_orchestrator_dependency() -> None:
    violations: list[str] = []
    for path in python_files("fusion"):
        for module in imported_modules(path):
            if module.startswith("cryoguard.orchestrator"):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == [], "\n".join(violations)


def test_the_composition_root_is_allowed_to_write_reports() -> None:
    """A guard against the opposite mistake: the pipeline must still wire the report."""

    modules = imported_modules(SRC / "pipeline.py")
    assert any(module.startswith("cryoguard.report") for module in modules)
