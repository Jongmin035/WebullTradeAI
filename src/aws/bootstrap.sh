#!/bin/bash
# Runs via cloud-init bootcmd on every EC2 boot.
# Installs config-pull on first boot; subsequent boots it's a no-op.
LOG=/var/log/webull-bootstrap.log
exec >> "$LOG" 2>&1
echo "[$(date)] bootstrap started"

if systemctl is-enabled config-pull.service >/dev/null 2>&1; then
    echo "[$(date)] config-pull already installed — nothing to do"
    exit 0
fi

echo "[$(date)] First-time setup: installing config-pull from S3"

ENV_FILE=/home/ec2-user/WebullTradeAI/.env
BUCKET=$(grep '^AWS_S3_BUCKET=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")

if [ -z "$BUCKET" ]; then
    echo "[$(date)] ERROR: AWS_S3_BUCKET not in .env — stopping bot.timer as fallback"
    systemctl stop bot.timer || true
    exit 1
fi

echo "[$(date)] Bucket: $BUCKET"

# Download startup.sh (the per-boot S3 config pull script)
aws s3 cp "s3://$BUCKET/config/startup.sh" /home/ec2-user/WebullTradeAI/src/aws/startup.sh
chmod +x /home/ec2-user/WebullTradeAI/src/aws/startup.sh
chown ec2-user:ec2-user /home/ec2-user/WebullTradeAI/src/aws/startup.sh

# Download and install config-pull.service
aws s3 cp "s3://$BUCKET/config/config-pull.service" /etc/systemd/system/config-pull.service
systemctl daemon-reload
systemctl enable config-pull.service

# Run config-pull now — installs the latest bot.service and retrain.service from S3
systemctl start config-pull.service

# Stop bot.timer just for this boot to prevent shutdown before retrain fires.
# On all future boots config-pull runs first (Before=bot.timer) so this isn't needed.
systemctl stop bot.timer || true

echo "[$(date)] bootstrap complete — retrain.timer will fire via Persistent=true"
