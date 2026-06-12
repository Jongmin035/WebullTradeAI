#!/bin/bash
# Stamp VERSION with the current git hash and upload source files to S3.
#
# Usage:
#   bash src/aws/deploy.sh                          # deploy all files
#   bash src/aws/deploy.sh src/core/trader.py       # deploy specific file(s)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GIT_SHA=$(git rev-parse --short HEAD)
BUCKET=$(grep '^AWS_S3_BUCKET=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
BUCKET=${BUCKET:-webull-trade-ai}

SOURCE_FILES=(
    src/aws/main.py
    src/core/trader.py
    src/core/predict.py
    src/aws/retrain.py
    src/aws/startup.sh
)

deploy_file() {
    local f="$1"
    # startup.sh lives at config/startup.sh; everything else goes under config/src/
    if [[ "$f" == "src/aws/startup.sh" ]]; then
        local s3_key="config/startup.sh"
    else
        local s3_key="config/src/${f#src/}"
    fi
    echo "  $f  ->  s3://$BUCKET/$s3_key  (v$GIT_SHA)"
    sed "s/__VERSION__/$GIT_SHA/g" "$f" | aws s3 cp - "s3://$BUCKET/$s3_key"
}

if [ $# -gt 0 ]; then
    echo "Deploying $# file(s)  git=$GIT_SHA"
    for f in "$@"; do deploy_file "$f"; done
else
    echo "Deploying all source files  git=$GIT_SHA"
    for f in "${SOURCE_FILES[@]}"; do deploy_file "$f"; done
fi

echo "Done."
