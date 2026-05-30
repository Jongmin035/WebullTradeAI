"""
One-time setup: enable S3 static website hosting and public read access
for dashboard files only (index.html, stats.json, history.html, performance_stats.json).

State files (state/*, data/*, models/*) remain private.

Run once on EC2 (IAM role handles auth) or locally if AWS CLI is configured:
    python src/aws/setup_public_dashboard.py
"""

import json
import os
import sys
from dotenv import load_dotenv

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

import boto3

BUCKET = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")
REGION = os.getenv("AWS_REGION", "us-east-1")

DASHBOARD_FILES = [
    "index.html",
    "stats.json",
    "history.html",
    "performance_stats.json",
]

BUCKET_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadDashboard",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": [f"arn:aws:s3:::{BUCKET}/{f}" for f in DASHBOARD_FILES],
        }
    ],
}


def main():
    s3  = boto3.client("s3", region_name=REGION)

    print(f"Configuring bucket: {BUCKET}")

    # 1. Disable Block Public Access (required before bucket policy takes effect)
    print("  Disabling Block Public Access...")
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls":       False,
            "IgnorePublicAcls":      False,
            "BlockPublicPolicy":     False,
            "RestrictPublicBuckets": False,
        },
    )

    # 2. Apply bucket policy (public read on dashboard files only)
    print("  Applying bucket policy...")
    s3.put_bucket_policy(
        Bucket=BUCKET,
        Policy=json.dumps(BUCKET_POLICY),
    )

    # 3. Enable static website hosting
    print("  Enabling static website hosting...")
    s3.put_bucket_website(
        Bucket=BUCKET,
        WebsiteConfiguration={"IndexDocument": {"Suffix": "index.html"}},
    )

    url = f"http://{BUCKET}.s3-website-{REGION}.amazonaws.com/"
    print(f"\nDone! Dashboard URL:\n  {url}")


if __name__ == "__main__":
    main()