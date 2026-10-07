#!/bin/bash

set -e

MSG="${1:-update}"

git add .

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "$MSG"
git push
