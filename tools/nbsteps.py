"""Turns src/pipeline.py::run() into readable notebook step cells (single source of truth: pipeline.py).

Markers inside run(), at the function's top indentation:
    # %% Title | one-line description          -> starts a new step (markdown heading + code cell)
    # show: <expression>                        -> display(<expression>) at the end of that step's cell
Code before the first marker becomes the "Start" step. `return x` inside run() stops the notebook with a message
(`_stop`), the final `return x` simply shows x. Relative imports become `src.` imports.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

PRELUDE = '''import sys, time
from IPython.display import display
{purge}import src.pipeline as _pipeline
globals().update({{k: v for k, v in vars(_pipeline).items() if not k.startswith("__")}})   # helpers + imports


class StopRun(Exception):
    """Raised on purpose (e.g. no GPU yet / paused): Run All again later to continue."""


def _stop(result):
    print(result)
    raise StopRun(result.get("status", "stopped") if isinstance(result, dict) else str(result))


{args}'''
PURGE = ('for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:\n'
         '    del sys.modules[_m]                       # always use the freshly written code (no stale imports)\n')


def _run_fn(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")


def _returns(fn):
    out = []

    def visit(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(ch, ast.Return):
                out.append(ch)
            visit(ch)
    visit(fn)
    return out


def steps(pipeline_py, purge: bool = True):
    """-> (prelude_code, [(title, description, code), ...])"""
    src = Path(pipeline_py).read_text(encoding="utf-8")
    fn = _run_fn(ast.parse(src))
    lines = src.splitlines()
    pos = fn.args.args[1:]
    defaults = fn.args.defaults
    args = "\n".join(f"{a.arg} = {ast.unparse(d)}" for a, d in zip(pos[len(pos) - len(defaults):], defaults))
    body = lines[fn.lineno:fn.end_lineno]                           # signature is one line
    last = fn.body[-1]
    for r in sorted(_returns(fn), key=lambda r: -r.lineno):         # bottom-up: insertions keep indices valid
        i0, i1 = r.lineno - 1 - fn.lineno, r.end_lineno - 1 - fn.lineno
        if r is last:                                               # final return -> result = ...; show it
            ind = re.match(r"^(\s*)", body[i0]).group(1)
            body[i0] = re.sub(r"^(\s*)return\s+", r"\1result = ", body[i0])
            body.insert(i1 + 1, ind + "result")
        else:
            body[i0] = re.sub(r"^(\s*)return\s+", r"\1_stop(", body[i0])
            body[i1] += ")"
    body = [re.sub(r"^(\s*)from \.(\w+)? import", lambda m: f"{m.group(1)}from src{'.' + m.group(2) if m.group(2) else ''} import", l)
            for l in body]
    body = textwrap.dedent("\n".join(body)).split("\n")
    out, title, desc, code, shows = [], "Start", "Logging, folders and the checkpoint store.", [], []

    def flush():
        c = "\n".join(code).strip("\n")
        if shows:
            c += "\n" + "\n".join(f"display({s})" for s in shows)
        if c.strip():
            out.append((title, desc, c))

    for l in body:
        m = re.match(r"^# %%\s*(.*)$", l)
        if m:
            flush()
            title, _, desc = (x.strip() for x in m.group(1).partition("|"))
            code, shows = [], []
            continue
        s = re.match(r"^# show:\s*(.+)$", l)
        if s:
            shows.append(s.group(1).strip())
            continue
        code.append(l)
    flush()
    return PRELUDE.format(purge=PURGE if purge else "", args=args).rstrip() + "\n", out


def cells(pipeline_py, first_num: int):
    """Notebook cells: a 'Load the pipeline' step + one markdown/code pair per step."""
    import nbformat as nbf
    prelude, st = steps(pipeline_py)
    cs = [nbf.v4.new_markdown_cell(f"## {first_num} · Load the pipeline\nImports the project code written above. "
                                   "The cells below are the steps of `src/pipeline.py::run()`, one per cell."),
          nbf.v4.new_code_cell(prelude)]
    for i, (t, d, c) in enumerate(st, first_num + 1):
        cs += [nbf.v4.new_markdown_cell(f"## {i} · {t}" + (f"\n{d}" if d else "")), nbf.v4.new_code_cell(c)]
    return cs, first_num + 1 + len(st)


def exec_steps(pipeline_py, cfg, **kwargs):
    """Runs the notebook steps in one namespace (used by the tests). Returns the namespace."""
    prelude, st = steps(pipeline_py, purge=False)
    ns = {"__name__": "__nb__"}
    exec(prelude, ns)
    ns.update(cfg=cfg, **kwargs)
    for t, _, c in st:
        exec(compile(c, f"<step {t}>", "exec"), ns)
    return ns
