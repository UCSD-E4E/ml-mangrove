"""
Pre-export preflight check — run locally before submitting GEE export tasks.

Verifies gcloud + EE authentication, inspects GCS bucket state per region,
and optionally clears prefixes before a fresh export.

Usage:
  # Auth check + bucket state for one region
  python preflight.py --bucket e4e-mangrove --region brazil

  # Check all regions at once
  python preflight.py --bucket e4e-mangrove --all-regions

  # Clear a region prefix before re-exporting (prompts for confirmation)
  python preflight.py --bucket e4e-mangrove --region brazil --clear
"""

import argparse
import subprocess
import sys
import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'regions.yaml')
EE_PROJECT  = 'e4e-mangrove'


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # shell=True required on Windows where gcloud/gsutil are .cmd files
    return subprocess.run(cmd, capture_output=True, text=True, shell=True)


def check_gcloud_auth() -> str | None:
    r = _run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'])
    account = r.stdout.strip()
    if not account:
        print('  gcloud account : NOT AUTHENTICATED')
        print('  → Run: gcloud auth login')
        return None
    print(f'  gcloud account : {account}')
    return account


def check_gcloud_project() -> str:
    r = _run(['gcloud', 'config', 'get-value', 'project'])
    project = r.stdout.strip() or '(not set)'
    print(f'  gcloud project : {project}')
    return project


def check_appdefault_auth() -> bool:
    r = _run(['gcloud', 'auth', 'application-default', 'print-access-token'])
    ok = r.returncode == 0
    status = 'OK' if ok else 'MISSING — run: gcloud auth application-default login'
    print(f'  app-default    : {status}')
    return ok


def list_prefix(bucket: str, prefix: str) -> tuple[int, int]:
    """Returns (file_count, total_bytes). Returns (-1, 0) on error."""
    r = _run(['gsutil', 'ls', '-l', f'gs://{bucket}/{prefix}/'])
    if r.returncode != 0:
        stderr = r.stderr.lower()
        if 'no urls matched' in stderr or 'matched no objects' in stderr:
            return 0, 0
        print(f'    gsutil error: {r.stderr.strip()}')
        return -1, 0

    total_bytes = 0
    file_count  = 0
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if not parts or parts[0].upper() == 'TOTAL:':
            continue
        try:
            total_bytes += int(parts[0])
            file_count  += 1
        except ValueError:
            pass
    return file_count, total_bytes


def clear_prefix(bucket: str, prefix: str) -> bool:
    r = _run(['gsutil', '-m', 'rm', '-r', f'gs://{bucket}/{prefix}/**'])
    return r.returncode == 0


def inspect_regions(bucket: str, region_keys: list[str], configs: dict) -> list[tuple[str, str]]:
    """Print bucket state for each region. Returns list of non-empty (name, prefix) pairs."""
    print(f'\n=== Bucket: gs://{bucket} ===')
    non_empty = []
    for name in region_keys:
        cfg    = configs[name]
        prefix = cfg['gcs_prefix']
        n, b   = list_prefix(bucket, prefix)
        if n < 0:
            status = 'ERROR'
        elif n == 0:
            status = 'empty'
        else:
            status = f'{n} files  ({b / 1e9:.2f} GB)'
            non_empty.append((name, prefix))
        print(f'  [{name}]  gs://{bucket}/{prefix}/  →  {status}')
    return non_empty


def main() -> None:
    parser = argparse.ArgumentParser(description='Pre-export GCP/EE preflight check.')
    parser.add_argument('--bucket',      required=True, help='GCS bucket name')
    parser.add_argument('--region',      default=None,  help='Single region key from regions.yaml')
    parser.add_argument('--all-regions', action='store_true', help='Inspect all regions')
    parser.add_argument('--clear',       action='store_true', help='Clear non-empty prefixes (prompts)')
    args = parser.parse_args()

    print('\n=== GCP / EE Preflight ===\n')
    account   = check_gcloud_auth()
    check_gcloud_project()
    check_appdefault_auth()
    print(f'  EE project     : {EE_PROJECT}')

    if account is None:
        sys.exit(1)

    if not args.region and not args.all_regions:
        print('\nNo region specified — auth check only.')
        print('Pass --region <name> or --all-regions to inspect bucket state.')
        return

    with open(CONFIG_PATH) as f:
        configs = yaml.safe_load(f)

    if args.all_regions:
        region_keys = list(configs.keys())
    else:
        if args.region not in configs:
            print(f'\nERROR: Unknown region "{args.region}". Available: {list(configs.keys())}')
            sys.exit(1)
        region_keys = [args.region]

    non_empty = inspect_regions(args.bucket, region_keys, configs)

    if not args.clear:
        if non_empty:
            regions_str = ' '.join(f'--region {n}' for n, _ in non_empty)
            print(f'\nTo clear before re-exporting, re-run with --clear [{regions_str}]')
        return

    if not non_empty:
        print('\nAll prefixes are empty — nothing to clear.')
        return

    print(f'\n{"!" * 55}')
    print('WARNING: This will permanently delete the files above.')
    print(f'{"!" * 55}')
    confirm = input('\nType "yes" to confirm deletion: ').strip().lower()
    if confirm != 'yes':
        print('Aborted.')
        return

    for name, prefix in non_empty:
        print(f'  Clearing gs://{args.bucket}/{prefix}/ ...', end=' ', flush=True)
        ok = clear_prefix(args.bucket, prefix)
        print('done' if ok else 'FAILED')


if __name__ == '__main__':
    main()
