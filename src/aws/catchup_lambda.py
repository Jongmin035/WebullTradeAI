"""
Outcome-based catch-up for bot.service.

Runs as a Lambda, triggered by a separate EventBridge schedule fired well
after bot.timer's normal target (e.g. 11:35 AM ET, 35 min after the 11:00 AM
slot). Checks whether today's run actually happened (diagnostics/{date}-bot-
exit.txt in S3) -- not systemd's calendar bookkeeping, which is exactly what
caused the Persistent=true duplicate-fire bug. If today's run is missing,
starts the EC2 instance (if needed) and invokes bot.service directly via SSM,
bypassing bot.timer/OnCalendar entirely. Idempotent: if the marker already
exists, this does nothing, so it can never double-trigger a day that already
ran.

Deploy: zip this file, `aws lambda create-function` with a Python 3.12+
runtime, handler `catchup_lambda.handler`, using the webull-bot-catchup-lambda
execution role. Wire an EventBridge Scheduler entry to invoke it on the
check schedule, with the input: {"job": "bot"}.
"""

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
from botocore.exceptions import ClientError

BUCKET      = "webull-trade-ai"
INSTANCE_ID = "i-0c931fbcbc7fb1d58"
REGION      = "us-east-1"
ET          = ZoneInfo("America/New_York")

SSM_WAIT_ATTEMPTS = 40   # 40 * 15s = 10 min max wait for the SSM agent to come online
SSM_WAIT_INTERVAL = 15

s3  = boto3.client("s3", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)


def _already_ran_today(job, today):
    if job == "bot":
        try:
            s3.head_object(Bucket=BUCKET, Key=f"diagnostics/{today}-bot-exit.txt")
            return True
        except ClientError as e:
            # Without s3:ListBucket, S3 returns a bare 403 (not 404) for a HeadObject
            # on a key that doesn't exist, to avoid confirming existence to an
            # unauthorized caller. Our role only has GetObject, so treat both as
            # "hasn't run yet" rather than granting ListBucket just to get a 404.
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "403", "Forbidden"):
                return False
            raise
    else:
        obj  = s3.get_object(Bucket=BUCKET, Key="retrain/last_run.json")
        data = json.loads(obj["Body"].read())
        return data.get("finished_at", "").startswith(today)


def _ensure_instance_running():
    state = ec2.describe_instances(InstanceIds=[INSTANCE_ID])["Reservations"][0]["Instances"][0]["State"]["Name"]
    if state == "stopped":
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
        print("Started EC2 instance")
    elif state not in ("running", "pending"):
        raise RuntimeError(f"Unexpected instance state: {state}")


def _wait_for_ssm_online():
    for _ in range(SSM_WAIT_ATTEMPTS):
        info = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [INSTANCE_ID]}]
        )
        rows = info.get("InstanceInformationList", [])
        if rows and rows[0]["PingStatus"] == "Online":
            return True
        time.sleep(SSM_WAIT_INTERVAL)
    return False


def handler(event, context):
    job     = event.get("job", "bot")
    dry_run = event.get("dry_run", False)
    today   = datetime.now(ET).strftime("%Y-%m-%d")

    ran = _already_ran_today(job, today)
    if ran:
        print(f"{job}: already ran today ({today}) — nothing to do")
        return {"triggered": False, "reason": "already_ran"}

    if dry_run:
        print(f"{job}: did NOT run today ({today}) — dry_run=True, stopping before any real action")
        return {"triggered": False, "reason": "dry_run_would_have_triggered"}

    print(f"{job}: did NOT run today ({today}) — triggering catch-up")

    _ensure_instance_running()

    if not _wait_for_ssm_online():
        print("SSM agent never came online within the wait window — giving up this attempt")
        return {"triggered": False, "reason": "ssm_timeout"}

    service = "bot.service" if job == "bot" else "retrain.service"
    ssm.send_command(
        InstanceIds=[INSTANCE_ID],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [f"systemctl start {service}"]},
    )
    print(f"Sent 'systemctl start {service}' via SSM")
    return {"triggered": True}
