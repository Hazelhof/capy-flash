#!/usr/bin/env bash
# clang-format for the C/C++/CUDA lines a change touches, not whole files (most of the tree predates
# .clang-format enforcement). As a pre-commit hook it formats the staged lines in place; under
# `pre-commit run --from-ref A --to-ref B` (CI) it checks the lines changed between A and B.
set -euo pipefail
extensions=c,cc,cpp,cu,cuh,h,hpp,inc
if [ -n "${PRE_COMMIT_FROM_REF:-}" ] && [ -n "${PRE_COMMIT_TO_REF:-}" ]; then
  diff=$(git-clang-format --extensions "$extensions" --diff "$PRE_COMMIT_FROM_REF" "$PRE_COMMIT_TO_REF" 2>&1 || true)
  if printf '%s\n' "$diff" | grep -q '^diff --git'; then
    printf '%s\n' "$diff"
    echo "clang-format: the changed lines above are not formatted; run pre-commit install and commit again" >&2
    exit 1
  fi
  exit 0
fi
output=$(git-clang-format --extensions "$extensions" --staged 2>&1 || true)
case $output in
  *"changed files"*) printf '%s\n' "$output"; exit 1 ;; # files reformatted: review, git add, commit again
esac
