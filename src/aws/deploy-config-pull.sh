#!/bin/bash
# ONE-TIME SETUP — run this in the EC2 Instance Connect browser terminal.
# After this, service file changes only require uploading to S3 (no SSH needed).
set -euo pipefail

cd /home/ec2-user/WebullTradeAI

# Stop bot.timer immediately so it can't trigger auto-shutdown while we work
sudo systemctl stop bot.timer
echo "bot.timer stopped"

# Pull latest code from git (gets startup.sh and config-pull.service)
git pull origin main
echo "git pull done"

# Make startup.sh executable
chmod +x src/aws/startup.sh

# Install the config-pull service
sudo cp src/aws/config-pull.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable config-pull.service
echo "config-pull.service installed and enabled"

# Run it now to confirm it works
sudo systemctl start config-pull.service
systemctl status config-pull.service --no-pager

echo ""
echo "Setup complete. To trigger today's retrain:"
echo "  sudo systemctl start retrain.service"
echo "  journalctl -u webull-retrain -f"
