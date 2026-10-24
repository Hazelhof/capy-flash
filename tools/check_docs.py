#!/usr/bin/env python3
"""Documentation guard: what the code accepts must be in the user docs.

  - every `ninfer-serve` flag parsed in src/serve/serve_options.cpp appears in the option tables of
    docs/serving.md (a table row naming it in backticks);
  - every NINFER_* environment variable read with getenv() under src/ appears in
    docs/capy-flash/diagnostics.md or docs/serving.md;
  - every conversion recipe in tools/convert/official_recipes.py appears in docs/weight-conversion.md;
  - every experiment tag (T<number>, e.g. T161 or T128b) cited in src/ or include/ has an entry in
    docs/capy-flash/development-log.md.

Usage: check_docs.py [repository root]. Exit 1 with the list of what is missing.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
read = lambda p: (root / p).read_text(encoding="utf-8")
sources = [p for d in ("src", "include") for p in (root / d).rglob("*")
           if p.suffix in {".cpp", ".h", ".cu", ".cuh", ".inc"}]
missing = []

serving = read("docs/serving.md")
table_rows = "\n".join(l for l in serving.splitlines() if l.startswith("|"))
flags = sorted(set(re.findall(r'arg == "(--[a-z0-9-]+)"', read("src/serve/serve_options.cpp"))))
missing += [f"serve flag {f} not in a docs/serving.md option table"
            for f in flags if f"`{f}`" not in table_rows and f"`{f} " not in table_rows]

env_docs = read("docs/capy-flash/diagnostics.md") + serving
envs = sorted({m for p in sources
               for m in re.findall(r'getenv\("(NINFER_[A-Z0-9_]+)"', p.read_text(errors="replace"))})
missing += [f"environment variable {e} not in docs/capy-flash/diagnostics.md or docs/serving.md"
            for e in envs if not re.search(rf"`{e}`", env_docs)]

recipes_src = read("tools/convert/official_recipes.py")
block = recipes_src[recipes_src.index("RECIPES = {"):]
recipes = re.findall(r'^\s+"([a-z0-9_]+)":', block[:block.index("}")], re.M)
conversion = read("docs/weight-conversion.md")
missing += [f"recipe {r} not in docs/weight-conversion.md" for r in recipes if f"`{r}`" not in conversion]

log = read("docs/capy-flash/development-log.md")
tags = sorted({m for p in sources for m in re.findall(r"\bT([1-9][0-9]{0,2}[a-z]?)\b", p.read_text(errors="replace"))},
              key=lambda t: (int(re.match(r"\d+", t).group()), t))
def logged(tag):
    n = int(re.match(r"\d+", tag).group())
    if re.search(rf"\bT{tag}\b", log):
        return True
    # ranges such as **T164-T166**
    return any(int(a) <= n <= int(b) for a, b in re.findall(r"\bT(\d+)-T(\d+)\b", log))
missing += [f"tag T{t} cited in the sources has no entry in docs/capy-flash/development-log.md"
            for t in tags if not logged(t)]

if missing:
    print("\n".join(missing))
    print(f"check_docs: {len(missing)} missing")
    sys.exit(1)
print(f"check_docs: {len(flags)} flags, {len(envs)} variables, {len(recipes)} recipes, {len(tags)} tags documented")
