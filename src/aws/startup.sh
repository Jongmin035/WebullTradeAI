#!/bin/bash
# Runs on every boot (via config-pull.service) before bot.timer / retrain.timer fire.
# Downloads the latest service files from S3 so changes made locally take effect
# without requiring SSH access to the instance.
set -euo pipefail

ENV_FILE=/home/ec2-user/WebullTradeAI/.env
BUCKET=$(grep '^AWS_S3_BUCKET=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")

if [ -z "$BUCKET" ]; then
    echo "config-pull: AWS_S3_BUCKET not found in .env — skipping"
    exit 0
fi

echo "config-pull: pulling service files from s3://$BUCKET/config/"

aws s3 cp "s3://$BUCKET/config/bot.service"     /etc/systemd/system/bot.service
aws s3 cp "s3://$BUCKET/config/retrain.service" /etc/systemd/system/retrain.service

systemctl daemon-reload
echo "config-pull: done"
