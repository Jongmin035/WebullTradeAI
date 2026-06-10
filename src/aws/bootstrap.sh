#!/bin/bash
# Runs via cloud-init bootcmd on every EC2 boot.
# Installs config-pull on first boot; refreshes startup.sh from S3 on every boot.
LOG=/var/log/webull-bootstrap.log
exec >> "$LOG" 2>&1
echo "[$(date)] bootstrap started"

ENV_FILE=/home/ec2-user/WebullTradeAI/.env
BUCKET=$(grep '^AWS_S3_BUCKET=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
export AWS_ACCESS_KEY_ID=$(grep '^AWS_ACCESS_KEY_ID=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
export AWS_SECRET_ACCESS_KEY=$(grep '^AWS_SECRET_ACCESS_KEY=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
export AWS_DEFAULT_REGION=$(grep '^AWS_REGION=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")

if [ -z "$BUCKET" ] || [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "[$(date)] ERROR: .env missing bucket or credentials — aborting"
    exit 1
fi

# Always refresh startup.sh from S3 so changes deploy without SSH.
STARTUP=/home/ec2-user/WebullTradeAI/src/aws/startup.sh
aws s3 cp "s3://$BUCKET/config/startup.sh" "$STARTUP" 2>/dev/null && \
    chmod +x "$STARTUP" && \
    chown ec2-user:ec2-user "$STARTUP" && \
    echo "[$(date)] startup.sh refreshed from S3" || \
    echo "[$(date)] startup.sh refresh skipped (S3 copy failed)"

if systemctl is-enabled config-pull.service >/dev/null 2>&1; then
    echo "[$(date)] config-pull already installed — nothing to do"
    exit 0
fi

echo "[$(date)] First-time setup: installing config-pull"

# Stop bot.timer now to prevent auto-shutdown during setup.
# On future boots, config-pull.service (Before=bot.timer) handles ordering.
systemctl stop bot.timer 2>/dev/null || true

echo "[$(date)] Downloading from S3"

aws s3 cp "s3://$BUCKET/config/config-pull.service" /etc/systemd/system/config-pull.service
systemctl daemon-reload
systemctl enable config-pull.service

# Run config-pull now to install the latest bot.service / retrain.service from S3
systemctl start config-pull.service

echo "[$(date)] bootstrap complete — retrain.timer will fire via Persistent=true"
