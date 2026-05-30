"""
Model artifact cleanup.

Removes saved model artifacts from local disk and/or S3.
Run this after a smoke test before starting the full training run,
or any time you want to reset the deployed model.

Usage:
    python src/aws/cleanup.py                      # delete local + S3 (all models)
    python src/aws/cleanup.py --local-only         # delete local only (keep S3)
    python src/aws/cleanup.py --s3-only            # delete S3 only (keep local)
    python src/aws/cleanup.py --model lstm          # delete one specific model only
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(SRC_DIR)
for d in (SRC_DIR, os.path.join(SRC_DIR, "core")):
    if d not in sys.path:
        sys.path.insert(0, d)

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

KNOWN_MODELS = ["lstm"]


def confirm(prompt):
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer == "y"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true", help="Delete local artifacts only")
    parser.add_argument("--s3-only",    action="store_true", help="Delete S3 artifacts only")
    parser.add_argument("--model",      type=str, choices=KNOWN_MODELS + [None],
                        default=None, metavar="MODEL",
                        help=f"Delete one model only. Choices: {KNOWN_MODELS}")
    parser.add_argument("--yes",        action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    scope      = f"model '{args.model}'" if args.model else "ALL models"
    targets    = []
    if not args.s3_only:
        targets.append("local disk")
    if not args.local_only:
        targets.append("S3")

    if not targets:
        log.error("--local-only and --s3-only cannot both be set.")
        sys.exit(1)

    print(f"\nThis will delete {scope} artifacts from: {' + '.join(targets)}")

    if not args.yes and not confirm("Proceed?"):
        print("Aborted.")
        sys.exit(0)

    from model_store import clean_local, clean_s3

    if not args.s3_only:
        clean_local(args.model)

    if not args.local_only:
        clean_s3(args.model)

    print("\nCleanup complete.")
    if not args.local_only:
        print("You must run retrain.py before main.py will work again.")
