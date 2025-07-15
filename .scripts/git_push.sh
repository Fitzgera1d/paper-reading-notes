#!/bin/bash

BASE_DIR=$(cd $(dirname $(realpath $0))/.. && pwd)

cd $BASE_DIR

python .scripts/render_readme.py
cp ./README_EN.md ./README.md

git add .
git commit -m "${1:-"update"}"
git push