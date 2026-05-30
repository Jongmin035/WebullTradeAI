#!/bin/bash
# Run once on EC2 after SSH-ing in.
# Usage: bash setup.sh

set -e

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install pandas numpy scikit-learn torch boto3 yfinance python-dotenv pyarrow matplotlib requests lxml

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Copy your .env file to the project root"
echo "  2. Run: python src/aws/train.py"
