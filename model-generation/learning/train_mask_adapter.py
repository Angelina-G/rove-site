"""Auditable training entry point for future Rove mask adaptation.

The default mode is a dry-run planner. Actual weight updates require a selected
segmentation backbone and a verified local training environment.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-samples', type=int, default=500)
    args = ap.parse_args()
    if args.min_samples < 1:
        ap.error('--min-samples 必须是正整数')
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if manifest.get('schemaVersion') != 1 or not isinstance(manifest.get('samples'), list):
        raise ValueError('manifest schema 无效')
    complete = int(manifest.get('completeCount', 0))
    if complete < 0 or complete > len(manifest['samples']):
        raise ValueError('completeCount 无效')
    status = 'ready-for-training' if complete >= args.min_samples else 'collect-more-confirmed-samples'
    plan = {
        'schemaVersion': 1,
        'status': status,
        'completeSamples': complete,
        'requiredSamples': args.min_samples,
        'nextStep': 'fine-tune a segmentation adapter with train/val split and held-out views' if status == 'ready-for-training' else 'continue recording confirmed source/corrected mask pairs',
        'weightsUpdated': False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'TRAINING_PLAN_OK status={status} complete={complete} output={out}')


if __name__ == '__main__':
    main()
