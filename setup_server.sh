#!/usr/bin/env bash
set -euo pipefail
repo="https://github.com/louppian/Oblique.git"
target="/shared/home/mai/Taehun/Uncertainty/Oblique"
command -v git >/dev/null
mkdir -p "$target"
cd "$target"
if [[ ! -e .git ]]; then
    git init -b main
fi
[[ "$(git rev-parse --show-toplevel)" == "$(pwd -P)" ]] || {
    echo "Repository root differs from $target" >&2; exit 1;
}
if git remote get-url origin >/dev/null 2>&1; then
    remote="$(git remote get-url origin)"
    case "$remote" in
        https://github.com/louppian/Oblique|https://github.com/louppian/Oblique.git|git@github.com:louppian/Oblique.git) ;;
        *) echo "Existing origin differs: $remote" >&2; exit 1 ;;
    esac
else
    git remote add origin "$repo"
fi
git fetch origin
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    [[ "$(git branch --show-current)" == "main" ]] || {
        echo "Select main before running setup." >&2; exit 1;
    }
    git merge --ff-only origin/main
else
    git checkout -B main --track origin/main
fi
git branch --set-upstream-to=origin/main main
mkdir -p cache ckpt data
git check-ignore cache/.ignore-check ckpt/.ignore-check data/.ignore-check
git status --short
printf '\nConnected: %s\n' "$target"
