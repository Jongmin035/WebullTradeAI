#!/bin/bash
# Runs on every boot via config-pull.service (Before=bot.timer / retrain.timer).
# Downloads the latest service files from S3, then pulls the latest Docker images
# from ECR so bot.service and retrain.service always run the newest code.
#
# Uses the EC2 IAM instance role (webull-bot-ec2) for AWS access — no .env credentials needed.
set -euo pipefail

BUCKET="webull-trade-ai"
export AWS_DEFAULT_REGION="us-east-1"

echo "config-pull: startup.sh started at $(date -u)" | \
    aws s3 cp - "s3://$BUCKET/diagnostics/$(date +%Y-%m-%d)-startup-begin.txt" || true

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

echo "config-pull: docker pull ok at $(date -u)" | \
    aws s3 cp - "s3://$BUCKET/diagnostics/$(date +%Y-%m-%d)-docker-pull-ok.txt" || true

# Self-update: keep this script current on the host so future changes take effect.
# Downloads the version from S3 (uploaded by CI or manually) for use on next boot.
aws s3 cp "s3://$BUCKET/config/startup.sh" "${0}.new" 2>/dev/null \
    && mv "${0}.new" "$0" \
    || true

echo "config-pull: done"
