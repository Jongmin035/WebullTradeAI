#!/bin/bash
# Runs on every boot via config-pull.service (Before=bot.timer / retrain.timer).
# Downloads the latest service files from S3, then pulls the latest Docker images
# from ECR so bot.service and retrain.service always run the newest code.
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
aws s3 cp "s3://$BUCKET/config/bot.timer"       /etc/systemd/system/bot.timer       || true
aws s3 cp "s3://$BUCKET/config/retrain.timer"   /etc/systemd/system/retrain.timer   || true

systemctl daemon-reload

# Pull latest Docker images from ECR so the next run uses the newest code.
# GitHub Actions pushes a new image on every git push to main.
echo "config-pull: logging in to ECR and pulling latest images"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR="${ACCOUNT}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"

aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | \
    docker login --username AWS --password-stdin "$ECR"

docker pull "${ECR}/webull-bot:latest"
docker tag  "${ECR}/webull-bot:latest" webull-bot:latest

docker pull "${ECR}/webull-retrain:latest"
docker tag  "${ECR}/webull-retrain:latest" webull-retrain:latest

echo "config-pull: done"
