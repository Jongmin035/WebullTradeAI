#!/bin/bash
# Runs on every boot via config-pull.service (Before=bot.timer / retrain.timer).
# Downloads the latest service files from S3 so changes take effect without SSH.
set -euo pipefail

ENV_FILE=/home/ec2-user/WebullTradeAI/.env
BUCKET=$(grep '^AWS_S3_BUCKET=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
export AWS_ACCESS_KEY_ID=$(grep '^AWS_ACCESS_KEY_ID=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
export AWS_SECRET_ACCESS_KEY=$(grep '^AWS_SECRET_ACCESS_KEY=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
export AWS_DEFAULT_REGION=$(grep '^AWS_REGION=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")

if [ -z "$BUCKET" ] || [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "config-pull: missing bucket or credentials in .env — skipping"
    exit 0
fi

echo "config-pull: pulling service files from s3://$BUCKET/config/"

aws s3 cp "s3://$BUCKET/config/bot.service"     /etc/systemd/system/bot.service
aws s3 cp "s3://$BUCKET/config/retrain.service" /etc/systemd/system/retrain.service

systemctl daemon-reload

# Pull latest Python source files if present in S3.
# Upload a file to s3://$BUCKET/config/src/ to deploy it on next boot.
CODE_DIR=/home/ec2-user/WebullTradeAI
echo "config-pull: syncing source files from s3://$BUCKET/config/src/ (if any)"
aws s3 sync "s3://$BUCKET/config/src/" "$CODE_DIR/src/" --quiet || true

echo "config-pull: done"
