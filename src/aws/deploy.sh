#!/bin/bash
# Upload infrastructure files to S3 so EC2 instances pick them up on next boot.
#
# Python source is no longer deployed via S3 — it's baked into Docker images
# by GitHub Actions on every git push to main. Only service/timer/startup files
# need to land in S3 so config-pull.service can refresh them at boot time.
#
# Usage:
#   bash src/aws/deploy.sh           # deploy all infra files
#   bash src/aws/deploy.sh src/aws/bot.service   # deploy a specific file

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BUCKET=$(grep '^AWS_S3_BUCKET=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
BUCKET=${BUCKET:-webull-trade-ai}

INFRA_FILES=(
    src/aws/startup.sh
    src/aws/bot.service
    src/aws/retrain.service
    src/aws/bot.timer
    src/aws/retrain.timer
)

deploy_file() {
    local f="$1"
    local s3_key="config/$(basename "$f")"
    echo "  $f  ->  s3://$BUCKET/$s3_key"
    aws s3 cp "$f" "s3://$BUCKET/$s3_key"
}

if [ $# -gt 0 ]; then
    echo "Deploying $# file(s)"
    for f in "$@"; do deploy_file "$f"; done
else
    echo "Deploying all infrastructure files"
    for f in "${INFRA_FILES[@]}"; do deploy_file "$f"; done
fi

echo "Done. Python source is deployed automatically via GitHub Actions on git push."
