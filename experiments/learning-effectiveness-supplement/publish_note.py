"""Copy the reviewed report and figures to the requested Obsidian location."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((HERE / 'prepared-manifest.json').read_text())
    target = Path(receipt['target'])
    if sha(target) != receipt['original_sha256']:
        raise ValueError('Obsidian source changed since report preparation; refusing to overwrite.')
    for filename, digest in receipt['files'].items():
        if sha(HERE / filename) != digest:
            raise ValueError(f'Prepared file changed: {filename}')
        destination = target.parent / filename
        if destination != target and destination.exists() and sha(destination) != digest:
            raise ValueError(f'Figure name already contains different data: {filename}')
    for filename, digest in receipt['files'].items():
        destination = target.parent / filename
        shutil.copyfile(HERE / filename, destination)
        if sha(destination) != digest:
            raise ValueError(f'Published file hash mismatch: {filename}')
    (HERE / 'published-manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(f'Updated report and two figures: {target}')
    print(f'Original note backup: {HERE / "original-note.md"}')


if __name__ == '__main__':
    main()
