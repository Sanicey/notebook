#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repository-wide CI Validator (.github/scripts/validate_repo.py)
与技能目录 `md_to_ipynb.py` 保持完全同源的 AST 顺序感知自包含校验、FP16 精确阈值检查与逐格隔离执行验收。
"""

import ast
import builtins
import json
import re
import symtable
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

FP16_MIN_FINITE = -65504.0


def _collect_target_names(target: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(_collect_target_names(elt))
    elif isinstance(target, ast.Starred):
        names.update(_collect_target_names(target.value))
    return names


def _check_toplevel_flow_undefined(tree: ast.Module, initial_defined: set[str]) -> set[str]:
    defined = set(initial_defined)
    missing: set[str] = set()

    def check_expr(node: ast.AST | None, local_env: set[str]):
        if node is None:
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in defined and node.id not in local_env:
                missing.add(node.id)
            return
        if isinstance(node, ast.NamedExpr):
            check_expr(node.value, local_env)
            bound = _collect_target_names(node.target)
            local_env.update(bound)
            defined.update(bound)
            return
        if isinstance(node, ast.Lambda):
            arg_names = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
            if node.args.vararg:
                arg_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                arg_names.add(node.args.kwarg.arg)
            for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                check_expr(d, local_env)
            check_expr(node.body, local_env | arg_names)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            comp_env = set(local_env)
            for gen in node.generators:
                check_expr(gen.iter, comp_env)
                comp_env.update(_collect_target_names(gen.target))
                for if_clause in gen.ifs:
                    check_expr(if_clause, comp_env)
            if isinstance(node, ast.DictComp):
                check_expr(node.key, comp_env)
                check_expr(node.value, comp_env)
            else:
                check_expr(node.elt, comp_env)
            return
        for child in ast.iter_child_nodes(node):
            check_expr(child, local_env)

    def check_stmts(stmts: list[ast.stmt]):
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                check_expr(stmt.value, set())
                for t in stmt.targets:
                    if not isinstance(t, ast.Name):
                        check_expr(t, set())
                    defined.update(_collect_target_names(t))
            elif isinstance(stmt, ast.AnnAssign):
                check_expr(stmt.annotation, set())
                if stmt.value is not None:
                    check_expr(stmt.value, set())
                    defined.update(_collect_target_names(stmt.target))
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    if stmt.target.id not in defined:
                        missing.add(stmt.target.id)
                else:
                    check_expr(stmt.target, set())
                check_expr(stmt.value, set())
                defined.update(_collect_target_names(stmt.target))
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound != "*":
                        defined.add(bound)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in stmt.decorator_list:
                    check_expr(dec, set())
                for d in stmt.args.defaults + [d for d in stmt.args.kw_defaults if d is not None]:
                    check_expr(d, set())
                defined.add(stmt.name)
            elif isinstance(stmt, ast.ClassDef):
                for dec in stmt.decorator_list:
                    check_expr(dec, set())
                for base in stmt.bases:
                    check_expr(base, set())
                for kw in stmt.keywords:
                    check_expr(kw.value, set())
                defined.add(stmt.name)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                check_expr(stmt.iter, set())
                defined.update(_collect_target_names(stmt.target))
                check_stmts(stmt.body)
                check_stmts(stmt.orelse)
            elif isinstance(stmt, ast.While):
                check_expr(stmt.test, set())
                check_stmts(stmt.body)
                check_stmts(stmt.orelse)
            elif isinstance(stmt, ast.If):
                check_expr(stmt.test, set())
                check_stmts(stmt.body)
                check_stmts(stmt.orelse)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    check_expr(item.context_expr, set())
                    if item.optional_vars is not None:
                        defined.update(_collect_target_names(item.optional_vars))
                check_stmts(stmt.body)
            elif isinstance(stmt, ast.Try):
                check_stmts(stmt.body)
                for handler in stmt.handlers:
                    if handler.type is not None:
                        check_expr(handler.type, set())
                    if handler.name:
                        defined.add(handler.name)
                    check_stmts(handler.body)
                check_stmts(stmt.orelse)
                check_stmts(stmt.finalbody)
            elif isinstance(stmt, ast.Delete):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        if t.id not in defined:
                            missing.add(t.id)
                        defined.discard(t.id)
                    else:
                        check_expr(t, set())
            else:
                check_expr(stmt, set())

    check_stmts(tree.body)
    return missing


def check_undefined(code: str, tree: ast.Module) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            return []
    st = symtable.symtable(code, "<cell>", "exec")
    base_builtins = set(dir(builtins)) | {"__name__", "__file__"}
    defined = set(base_builtins)
    referenced = set()

    def walk(tbl, is_top):
        for sym in tbl.get_symbols():
            name = sym.get_name()
            if is_top and (sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_namespace()):
                defined.add(name)
            if sym.is_global() and not sym.is_assigned() and not sym.is_imported() and sym.is_referenced():
                referenced.add(name)
        for child in tbl.get_children():
            walk(child, False)

    walk(st, True)
    return sorted((referenced - defined) | _check_toplevel_flow_undefined(tree, base_builtins))


def has_unsafe_mask(tree: ast.AST) -> bool:
    def eval_num(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = eval_num(node.operand)
            return -v if v is not None else None
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("masked_fill", "masked_fill_", "where"):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    num = eval_num(arg)
                    if num is not None and num < FP16_MIN_FINITE:
                        return True
    return False


def main() -> int:
    root = Path(".")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "<!-- STATS_START -->" in readme and "<!-- STATS_END -->" in readme, "Missing STATS anchors in README.md"
    assert "<!-- PAPER_TABLE_START -->" in readme and "<!-- PAPER_TABLE_END -->" in readme, (
        "Missing PAPER_TABLE_START / PAPER_TABLE_END anchors in README.md"
    )
    table_body = readme.split("<!-- PAPER_TABLE_START -->", 1)[1].split("<!-- PAPER_TABLE_END -->", 1)[0]
    table_lines = [l for l in table_body.strip().splitlines() if l.strip()]
    assert len(table_lines) >= 3, "README table must have header, separator, and >=1 row"
    for line in table_lines:
        assert line.strip().startswith("|"), f"Non-table line inside GFM table: {line!r}"

    indexed_notebooks = set()
    for row in table_lines[2:]:
        m = re.search(r"\(([^)]+/notebook\.ipynb)\)", row)
        assert m, f"Missing notebook link in row: {row}"
        rel_path = m.group(1)
        assert (root / rel_path).exists(), f"Linked notebook does not exist: {rel_path}"
        indexed_notebooks.add(Path(rel_path).as_posix())

    notebooks = sorted(root.glob("*/notebook.ipynb"))
    assert notebooks, "No paper notebooks found"
    disk_notebooks = {nb.as_posix() for nb in notebooks}
    unindexed = disk_notebooks - indexed_notebooks
    assert not unindexed, f"Found unindexed orphan notebooks on disk: {sorted(unindexed)}"

    for nb_path in notebooks:
        print(f"=== Validating & Executing {nb_path} ===")
        nb_dict = json.loads(nb_path.read_text(encoding="utf-8"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            nb_obj = nbformat.from_dict(nb_dict)
            nbformat.validate(nb_obj)
        assert not caught, f"nbformat warnings in {nb_path}: {caught}"

        for idx, cell in enumerate(nb_dict["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            for line in code.splitlines():
                assert not re.match(r"^ {0,3}(`{3,}|~{3,})", line), (
                    f"{nb_path} cell {idx} contains stray markdown fence: {line!r}"
                )
            tree = ast.parse(code)
            assert not has_unsafe_mask(tree), (
                f"{nb_path} cell {idx} uses unsafe FP16 mask constant (< {FP16_MIN_FINITE})"
            )
            missing = check_undefined(code, tree)
            assert not missing, f"{nb_path} cell {idx} not self-contained, missing/use-before-def: {missing}"
            plt.close("all")
            exec(compile(code, f"{nb_path}:cell-{idx}", "exec"), {"__name__": "__main__"})
            plt.close("all")
            print(f"  [PASS] cell {idx} ({cell.get('id')})")

    print("All notebooks validated and executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
