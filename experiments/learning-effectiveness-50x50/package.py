"""Verify preserved evidence and package all new outputs except local model weights."""
import io
import json
from pathlib import Path
import sys
import tarfile

from common import ROOT,HERE,config,dump,identity,sha,now


def read(path):
    return json.loads(Path(path).read_text())


def preservation():
    pinned=read(HERE/'preserved-inputs.json')['files']
    changed=[name for name,digest in pinned.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
    if changed:
        raise ValueError(f'Old evidence changed: {changed[:10]}')
    return len(pinned)


def archive(root):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.pt','.tmp')
                 and p.name not in ('.suite.lock', '.lock') and '__pycache__' not in p.parts)
    hashes={str(p.relative_to(root)):sha(p) for p in files}
    payload=json.dumps(hashes,indent=2).encode()
    with tarfile.open(HERE/'evidence.tar.gz','w:gz') as stream:
        for path in files:
            stream.add(path,arcname=str(path.relative_to(root)),recursive=False)
        info=tarfile.TarInfo('SHA256.json'); info.size=len(payload)
        stream.addfile(info,io.BytesIO(payload))
    import hashlib
    with tarfile.open(HERE/'evidence.tar.gz','r:gz') as stream:
        embedded=json.load(stream.extractfile('SHA256.json'))
        members=[m for m in stream.getmembers() if m.isfile() and m.name!='SHA256.json']
        if len(members)!=len(hashes) or embedded!=hashes:
            raise ValueError('Archive file list differs from evidence.')
        for member in members:
            if hashlib.sha256(stream.extractfile(member).read()).hexdigest()!=hashes[member.name]:
                raise ValueError(f'Archive hash mismatch: {member.name}')
    dump(HERE/'archive-validation.json',dict(files=len(files),all_hashes_match=True,
        archive_sha256=sha(HERE/'evidence.tar.gz'),models_excluded=True))


def main():
    cfg=config(); root=Path(cfg['runtime']['output_dir']); fingerprint,_=identity(cfg)
    training,evaluation=read(root/'experiment.json'),read(root/'evaluation.json')
    if training['fingerprint']!=fingerprint or training['status']!='complete' or evaluation['status']!='complete':
        raise ValueError('The full experiment has not completed unchanged.')
    for path,digest in evaluation['sources'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError(f'Evaluator changed: {path}')
    for row in evaluation['training_validation']['checks']:
        folder=root/'training'/row['group']/row['circuit']/f'seed-{row["seed"]}'
        for episode,digest in row['snapshot_sha256'].items():
            if sha(folder/f'snapshots/{episode}.pt')!=digest:
                raise ValueError('An evaluated training snapshot changed.')
        if sha(folder/'checkpoint.pt')!=row['snapshot_sha256'][str(cfg['protocol']['episodes'])]:
            raise ValueError('The final training checkpoint changed.')
    tests=read(HERE/'test-results.json')
    if not tests['successful'] or tests['skipped']:
        raise ValueError('Tests did not pass without skips.')
    for name,digest in tests['source_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'Tested source changed: {name}')
    if sha(HERE/'tests.log')!=tests['log_sha256']:
        raise ValueError('Test log changed.')
    validation=read(HERE/'log-validation.json')
    if validation['status']!='complete' or validation['committed_training_steps']!=67500 or validation['physical_evaluation_rollouts']!=900:
        raise ValueError('Missing full raw-log replay validation.')
    netlists=read(HERE/'netlist-validation.json')
    if netlists['status']!='complete' or netlists['exports_checked']!=930 or netlists['netlists_verified']!=1857:
        raise ValueError('Missing full exported-netlist CEC evidence.')
    old_count=preservation()
    dump(HERE/'preservation-validation.json',dict(time=now(),files_checked=old_count,all_hashes_match=True))
    models=read(HERE/'model-manifest.json')
    for name,digest in models['hashes'].items():
        if sha(root/name)!=digest:
            raise ValueError('Model weights changed after CEC.')
    archive(root)
    plot=read(HERE/'plot-validation.json')
    for name,digest in plot['hashes'].items():
        if sha(HERE/name)!=digest:
            raise ValueError('Plot input/output changed.')
    files=sorted(p for p in HERE.iterdir() if p.is_file() and p.name!='artifact-manifest.json')
    dump(HERE/'artifact-manifest.json',dict(command=[sys.executable,'-B',str(Path(__file__).resolve())],
        packaged_at=now(),training_fingerprint=fingerprint,training_runs=27,evaluation_physical_banks=30,
        evaluation_physical_rollouts=900,evaluation_logical_model_rows=36,evaluation_logical_rollouts=1080,
        prefixes=[5,10,20,30,40,50],netlists_verified=1857,models_local=True,
        old_files_unchanged=old_count,tests_passed=tests['tests_run'],original_tests_passed=15,
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}))
    print(f'Packaged {len(files)} artifacts; {old_count} old files unchanged.',flush=True)


if __name__=='__main__':
    main()
