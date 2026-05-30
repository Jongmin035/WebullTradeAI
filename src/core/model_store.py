"""
Model artifact storage.

Saves and loads trained model artifacts to/from local disk and S3.

Local layout:  src/models/artifacts/{model_name}/
S3 layout:     s3://{bucket}/models/{model_name}/

Supported model names: lstm
Metadata file:         src/models/artifacts/metadata.json
"""

import json
import os
import sys
import logging
from datetime import datetime

import joblib

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

log = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(_src, "models", "artifacts")
METADATA_FILE = os.path.join(ARTIFACTS_DIR, "metadata.json")


# --- Helpers ---

def _model_dir(model_name):
    d = os.path.join(ARTIFACTS_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return d


def _s3():
    import boto3
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_src, "..", ".env"))
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _bucket():
    import os
    return os.getenv("AWS_S3_BUCKET", "webull-trade-ai")


# --- Save ---

def save_artifacts(model_name, artifacts):
    """
    Save model artifacts dict to local disk.

    artifacts keys and expected types:
      lstm       : clf_state_dict, clf_scaler, reg_state_dict, reg_scaler (dict + sklearn)
    """
    import torch
    d = _model_dir(model_name)
    for name, obj in artifacts.items():
        if name.endswith("_state_dict"):
            torch.save(obj, os.path.join(d, f"{name}.pt"))
        else:
            joblib.dump(obj, os.path.join(d, f"{name}.pkl"))
    log.info(f"Saved {len(artifacts)} artifact(s) for '{model_name}' to {d}")


def save_metadata(winner, sharpe_scores, trained_up_to, evaluation_months=12):
    """Save metadata.json recording which model won and evaluation results."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    meta = {
        "winner":             winner,
        "trained_up_to":      str(trained_up_to),
        "evaluation_months":  evaluation_months,
        "sharpe_scores":      sharpe_scores,
        "retrained_at":       datetime.now().isoformat(timespec="seconds"),
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved — winner: {winner}  sharpes: {sharpe_scores}")
    return meta


# --- Load ---

def load_artifacts(model_name):
    """
    Load model artifacts from local disk. Downloads from S3 if missing.
    Returns dict of name -> object.
    """
    import torch
    d = _model_dir(model_name)
    _ensure_artifacts_local(model_name, d)

    artifacts = {}
    for fname in os.listdir(d):
        name = fname.rsplit(".", 1)[0]
        path = os.path.join(d, fname)
        if fname.endswith(".pt"):
            artifacts[name] = torch.load(path, map_location="cpu", weights_only=True)
        elif fname.endswith(".pkl"):
            artifacts[name] = joblib.load(path)
    return artifacts


def load_metadata():
    """Load metadata.json. Downloads from S3 if missing. Returns None if not found."""
    if not os.path.exists(METADATA_FILE):
        _download_file_from_s3("models/metadata.json", METADATA_FILE)
    if not os.path.exists(METADATA_FILE):
        return None
    with open(METADATA_FILE) as f:
        return json.load(f)


# --- S3 sync ---

def _ensure_artifacts_local(model_name, local_dir):
    """Download model artifacts from S3 if the local directory is empty."""
    if os.listdir(local_dir):
        return  # already have local copies
    try:
        s3 = _s3()
        bucket = _bucket()
        prefix = f"models/{model_name}/"
        paginator = s3.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
        ]
        for key in keys:
            local_path = os.path.join(local_dir, os.path.basename(key))
            s3.download_file(bucket, key, local_path)
        if keys:
            log.info(f"Downloaded {len(keys)} artifact(s) for '{model_name}' from S3")
    except Exception as e:
        log.warning(f"Could not download artifacts for '{model_name}' from S3: {e}")


def _download_file_from_s3(s3_key, local_path):
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _s3().download_file(_bucket(), s3_key, local_path)
    except Exception as e:
        log.warning(f"Could not download {s3_key} from S3: {e}")


def upload_artifacts_to_s3(model_name):
    """Upload all local artifacts for a model to S3."""
    try:
        s3 = _s3()
        bucket = _bucket()
        d = _model_dir(model_name)
        for fname in os.listdir(d):
            s3_key = f"models/{model_name}/{fname}"
            s3.upload_file(os.path.join(d, fname), bucket, s3_key)
        log.info(f"Uploaded artifacts for '{model_name}' to s3://{bucket}/models/{model_name}/")
    except Exception as e:
        log.warning(f"S3 upload failed for '{model_name}': {e}")


def upload_metadata_to_s3():
    try:
        _s3().upload_file(METADATA_FILE, _bucket(), "models/metadata.json")
        log.info("Metadata uploaded to S3")
    except Exception as e:
        log.warning(f"Metadata S3 upload failed: {e}")


# --- Cleanup ---

def clean_local(model_name=None):
    """
    Delete local model artifacts.
    If model_name is None, deletes everything under ARTIFACTS_DIR.
    """
    import shutil
    if model_name:
        target = os.path.join(ARTIFACTS_DIR, model_name)
        if os.path.exists(target):
            shutil.rmtree(target)
            log.info(f"Deleted local artifacts: {target}")
        else:
            log.info(f"No local artifacts found for '{model_name}'")
    else:
        if os.path.exists(ARTIFACTS_DIR):
            shutil.rmtree(ARTIFACTS_DIR)
            log.info(f"Deleted all local artifacts: {ARTIFACTS_DIR}")
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
            log.info("Deleted local metadata.json")


def clean_s3(model_name=None):
    """
    Delete model artifacts from S3.
    If model_name is None, deletes everything under s3://{bucket}/models/
    """
    try:
        s3     = _s3()
        bucket = _bucket()
        prefix = f"models/{model_name}/" if model_name else "models/"

        paginator = s3.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
        ]
        if not keys:
            log.info(f"No S3 objects found under s3://{bucket}/{prefix}")
            return

        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in keys]},
        )
        log.info(f"Deleted {len(keys)} object(s) from s3://{bucket}/{prefix}")
    except Exception as e:
        log.warning(f"S3 cleanup failed: {e}")
