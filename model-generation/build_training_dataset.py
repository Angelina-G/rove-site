"""Build an auditable manifest from confirmed mask correction records.

This intentionally does not train weights. It creates a reproducible dataset index;
records without both source and corrected masks remain visible as incomplete.
"""
import argparse
import json
import re
from pathlib import Path
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feedback-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    feedback_dir = Path(args.feedback_dir).resolve()
    image_root = feedback_dir.parent / 'test-images'
    rows = []
    seen = set()

    def local_mask(url, version, image_id):
        pattern = rf'^/api/qc/files/{re.escape(version)}/{image_id}-mask\\.png$'
        if not isinstance(url, str) or not re.fullmatch(pattern, url):
            return None
        path = image_root / version / f'{image_id}-mask.png'
        if not path.is_file():
            return None
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                if img.width * img.height > 25_000_000 or img.width < 2 or img.height < 2:
                    return None
        except Exception:
            return None
        return path

    for path in sorted(feedback_dir.glob('*.json')):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if row.get('confirmed') is False:
            continue
        image_id = row.get('id')
        angle = row.get('angle')
        key = (image_id, angle)
        if not re.fullmatch(r'[0-9a-f]{32}', image_id or '') or angle not in {'front', 'left', 'right', 'back'} or key in seen:
            continue
        source = local_mask(row.get('sourceMaskUrl'), 'processed-v2', image_id)
        corrected = local_mask(row.get('correctedMaskUrl'), 'processed-v3', image_id)
        if source and corrected:
            with Image.open(source) as a, Image.open(corrected) as b:
                if a.size != b.size:
                    continue
        seen.add(key)
        rows.append({
            'feedbackId': row.get('feedbackId', path.stem),
            'id': image_id,
            'angle': angle,
            'sourceMaskUrl': row.get('sourceMaskUrl'),
            'correctedMaskUrl': row.get('correctedMaskUrl'),
            'complete': bool(source and corrected),
            'createdAt': row.get('createdAt'),
        })
    complete = sum(1 for r in rows if r['complete'])
    payload = {
        'schemaVersion': 1,
        'purpose': 'Rove 人体/布料分割人工修正数据集索引',
        'sampleCount': len(rows),
        'completeCount': complete,
        'samples': rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'DATASET_MANIFEST_OK samples={len(rows)} complete={complete} output={out}')


if __name__ == '__main__':
    main()
