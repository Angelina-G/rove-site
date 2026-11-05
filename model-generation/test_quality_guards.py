import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


def main():
    td = ROOT / 'output' / 'test-quality-guards'
    td.mkdir(parents=True, exist_ok=True)
    jobs = td / 'jobs'; jobs.mkdir(exist_ok=True)
    out = td / 'comparison'
    result = run(str(ROOT / 'compose_comparison.py'), str(jobs), str(out))
    if result.returncode == 0:
        raise AssertionError('empty comparison input must fail')

    manifest = td / 'manifest.json'
    manifest.write_text(json.dumps({'schemaVersion': 1, 'completeCount': 0, 'samples': []}), encoding='utf-8')
    plan = td / 'plan.json'
    result = run(str(ROOT / 'learning' / 'train_mask_adapter.py'), '--manifest', str(manifest), '--output', str(plan), '--min-samples', '-1')
    if result.returncode == 0:
        raise AssertionError('negative min-samples must fail')
    print('quality guard tests passed')


if __name__ == '__main__':
    main()
