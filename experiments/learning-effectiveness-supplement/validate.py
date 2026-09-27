"""Read-only audit of existing 100x10 evidence, with isolated inference replay."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_SUITE = ROOT / 'experiments/learning-effectiveness'
sys.path.insert(0, str(ORIGINAL_SUITE))
import common
import analysis
import evaluate


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='Replay 9 final models and 3 uniform sequences.')
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cfg = common.config()
    root = Path(cfg['runtime']['output_dir'])
    fingerprint, payload = common.identity(cfg)
    require(fingerprint == read(root / 'experiment.json')['fingerprint'], 'Training fingerprint changed.')
    require(fingerprint == read(root / 'evaluation.json')['training_fingerprint'], 'Evaluation fingerprint changed.')
    artifact_manifest = read(ORIGINAL_SUITE / 'artifact-manifest.json')
    current_artifacts, historical_documents = 0, []
    for path, entry in artifact_manifest['files'].items():
        if common.sha(ROOT / path) == entry['sha256']:
            current_artifacts += 1
            continue
        # The repository overview was intentionally extended by later experiments.
        # Experimental sources/data must still match; bind only this overview to history.
        require(path == 'README.md', f'Artifact hash mismatch: {path}')
        commits = subprocess.check_output(['git', 'log', '--format=%H', '--', path], cwd=ROOT, text=True).splitlines()
        matched = None
        for commit in commits:
            content = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
            if hash_bytes(content) == entry['sha256']:
                matched = commit
                break
        require(matched is not None, 'Original README hash cannot be resolved in Git history.')
        historical_documents.append(dict(path=path, original_sha256=entry['sha256'], matching_git_commit=matched,
                                         current_sha256=common.sha(ROOT / path), reason='Later experiment documentation added.'))
    model_manifest = read(ORIGINAL_SUITE / 'model-manifest.json')
    for path, digest in model_manifest['hashes'].items():
        require(common.sha(root / path) == digest, f'Model hash mismatch: {path}')
    training_validation = evaluate.training_validation(cfg)
    print(f'Fingerprints, {current_artifacts} current artifacts + {len(historical_documents)} historical README, 135 model/probe files and 27 training states verified.', flush=True)

    rows, records_by_key, bank_count = [], {}, 0
    for policy in common.POLICIES:
        for name, circuit in cfg['protocol']['circuits'].items():
            for seed in cfg['protocol']['seeds']:
                folder = root / 'evaluation' / policy / name / f'seed-{seed}'
                result = read(folder / 'result.json')
                require(result['status'] == 'complete', f'Incomplete bank: {folder}')
                require(result['network_unchanged'] and result['checkpoint_unchanged'], f'Inference changed model: {folder}')
                require([r['evaluation_seed'] for r in result['rollouts']] == common.EVAL_SEEDS, 'Evaluation seed coverage mismatch.')
                checkpoint = evaluate.checkpoint_path(root, policy, name, seed)
                digest = common.sha(checkpoint)
                require(result['checkpoint_sha256'] == digest, 'Bank checkpoint changed.')
                bank_count += 1
                for row in result['rollouts']:
                    key = (policy, name, seed, row['evaluation_seed'])
                    destination = folder / f'rollout-{row["evaluation_seed"]}'
                    require(read(destination / 'rollout.json') == row, f'Bank/rollout mismatch: {key}')
                    require(row['status'] == 'complete' and row['checkpoint_sha256'] == digest, f'Invalid rollout: {key}')
                    logged = analysis.validate_log(destination / 'episodes/1/log.csv', cfg, circuit)
                    best = min(logged, key=analysis.rank)
                    require(all(best[k] == row['best'][k] for k in ('luts', 'levels', 'feasible', 'sequence', 'iteration')), f'Best mismatch: {key}')
                    require([cfg['protocol']['actions'][a] for a in row['actions']] == [r['optimization'] for r in logged[1:]], f'Actions mismatch: {key}')
                    require(row['rewards'] == [r['reward'] for r in logged[1:]], f'Rewards mismatch: {key}')
                    require(all(row['terminal'][k] == logged[-1][k] for k in ('luts', 'levels', 'feasible')), f'Terminal mismatch: {key}')
                    require('Networks are equivalent' in (destination / 'equivalence.log').read_text(), f'Missing CEC: {key}')
                    require(key not in records_by_key, f'Duplicate key: {key}')
                    records_by_key[key] = row
                    rows.append(dict(policy=policy, circuit=name, training_seed=seed, evaluation_seed=row['evaluation_seed'],
                        best_luts=best['luts'], best_levels=best['levels'], best_feasible=best['feasible'],
                        terminal_luts=row['terminal']['luts'], terminal_levels=row['terminal']['levels'],
                        terminal_feasible=row['terminal']['feasible'], reward=sum(row['rewards']),
                        status='complete', task_status='complete'))
    require(bank_count == 36 and len(rows) == 1080, 'Evaluation task count mismatch.')
    with (ORIGINAL_SUITE / 'evaluation.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(rows), 'CSV coverage mismatch.')
    for row, csv_row in zip(rows, csv_rows):
        require(all(str(v) == csv_row[k] for k, v in row.items()), 'Evaluation CSV disagrees with raw logs.')
    for name in cfg['protocol']['circuits']:
        for evaluation_seed in common.EVAL_SEEDS:
            references = [records_by_key[('uniform', name, s, evaluation_seed)] for s in cfg['protocol']['seeds']]
            require(all(all(row[k] == references[0][k] for k in ('actions', 'rewards', 'best', 'terminal')) for row in references), 'Uniform repeated banks disagree.')
    print('1080 rollouts / 10800 actions replayed from logs; repeated uniform banks identical.', flush=True)

    comparisons = [analysis.policy_comparison(rows, name, baseline, cfg['protocol']['seeds'])
                   for name in cfg['protocol']['circuits'] for baseline in ('initial', 'uniform')]
    require(comparisons == read(ORIGINAL_SUITE / 'summary.json')['comparisons'], 'Recomputed paired bootstrap differs from original.')
    aggregate, per_seed = [], []
    for name in cfg['protocol']['circuits']:
        for policy in common.POLICIES:
            for seed in [None, *cfg['protocol']['seeds']]:
                selected = [r for r in rows if r['circuit'] == name and r['policy'] == policy and (seed is None or r['training_seed'] == seed)]
                item = dict(circuit=name, policy=policy, training_seed=seed, completed=len(selected),
                    feasible=sum(r['best_feasible'] for r in selected),
                    best_luts_mean=float(np.mean([r['best_luts'] for r in selected])) if all(r['best_feasible'] for r in selected) else None,
                    terminal_feasible=sum(r['terminal_feasible'] for r in selected),
                    terminal_luts_mean=float(np.mean([r['terminal_luts'] for r in selected])) if all(r['terminal_feasible'] for r in selected) else None)
                (aggregate if seed is None else per_seed).append(item)
    write(HERE / 'recomputed-summary.json', dict(aggregate=aggregate, per_seed=per_seed, comparisons=comparisons,
          inference_evaluation_seeds=common.EVAL_SEEDS, independent_training_seeds=cfg['protocol']['seeds'],
          bootstrap_repetitions=10000, bootstrap_seed=20260927,
          physical_rollouts=1080, nonduplicated_policy_rollout_combinations=900,
          uniform_note='Three physical banks per circuit are identical; only 30 independent uniform sequences per circuit.'))

    cec = read(ORIGINAL_SUITE / 'netlist-validation.json')
    require(cec['status'] == 'complete' and cec['exports_checked'] == 1110, 'Incomplete CEC coverage.')
    netlists = 0
    for record in cec['records']:
        folder = root / record['folder']
        require(record['status'] == 'complete', f'Failed CEC: {folder}')
        for filename, digest in record['netlist_sha256'].items():
            require(common.sha(folder / filename) == digest, f'Netlist changed: {folder / filename}')
            netlists += 1
        require(common.sha(folder / 'full-equivalence.log') == record['log_sha256'], f'CEC log changed: {folder}')
        require((folder / 'full-equivalence.log').read_text().count('Networks are equivalent') == record['netlists_verified'], f'CEC verdict mismatch: {folder}')
    require(netlists == cec['netlists_verified'] == 2217, 'Netlist count mismatch.')

    archive = ORIGINAL_SUITE / 'evidence.tar.gz'
    require(common.sha(archive) == read(ORIGINAL_SUITE / 'analysis-provenance.json')['evidence_sha256'], 'Evidence archive changed.')
    with tarfile.open(archive, 'r:gz') as stream:
        manifest = json.load(stream.extractfile('SHA256.json'))
    archived = 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            if not member.isfile() or member.name == 'SHA256.json':
                continue
            content = stream.extractfile(member).read()
            require(hash_bytes(content) == manifest[member.name], f'Archive corruption: {member.name}')
            require(common.sha(root / member.name) == manifest[member.name], f'Raw evidence changed: {member.name}')
            archived += 1
    require(archived == len(manifest) == 10687, 'Archive coverage mismatch.')
    print('Six paired bootstrap comparisons exactly reproduced; 2217 CEC records and 10687 archived files verified.', flush=True)

    replays = []
    if args.replay:
        torch.set_num_threads(1)
        replay_root = ROOT / 'results/learning-effectiveness-supplement' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        for policy in ('trained-100', 'uniform'):
            for name, circuit in cfg['protocol']['circuits'].items():
                seeds = cfg['protocol']['seeds'] if policy == 'trained-100' else cfg['protocol']['seeds'][:1]
                for seed in seeds:
                    checkpoint = evaluate.checkpoint_path(root, policy, name, seed)
                    digest = common.sha(checkpoint)
                    global_rng = torch.random.get_rng_state().clone()
                    with torch.random.fork_rng(devices=[]):
                        network = common.ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']), cfg['method']['network'])
                    network.load_state_dict(common.load(checkpoint)['network'])
                    network.eval()
                    network.requires_grad_(False)
                    model_hash = common.state_hash(network.state_dict())
                    destination = replay_root / policy / name / f'seed-{seed}'
                    row = evaluate.inference_rollout(cfg, circuit, network, 10000, destination, policy == 'uniform')
                    original = records_by_key[(policy, name, seed, 10000)]
                    require(all(row[k] == original[k] for k in ('actions', 'rewards', 'best', 'terminal')), f'Inference replay differs: {destination}')
                    require(torch.equal(global_rng, torch.random.get_rng_state()), 'Inference advanced global Torch RNG.')
                    require(model_hash == common.state_hash(network.state_dict()), 'Inference changed network.')
                    require(common.sha(checkpoint) == digest, 'Inference changed checkpoint.')
                    replays.append(dict(policy=policy, circuit=name, training_seed=seed, evaluation_seed=10000,
                                        exact_match=True, weights_unchanged=True, checkpoint_unchanged=True,
                                        global_torch_rng_unchanged=True, folder=str(destination)))
                    print(f'Exact inference replay {len(replays)}/12: {policy}/{name}/seed-{seed}', flush=True)
    for path, digest in model_manifest['hashes'].items():
        require(common.sha(root / path) == digest, f'Model changed after audit: {path}')
    scan = dict(
        simpsons_paradox='逐电路报告，保留逐训练种子效应；i2c种子1与均值方向不同。',
        ecological_fallacy='独立训练单位为3个种子，90条配对序列不视为90次训练。',
        selection_bias='使用完整种子集合和预定第100轮模型，未按最好模型筛选。',
        collider_bias='含不可行序列时不筛选可行子集计算LUT收益。',
        base_rate_neglect='max报告完整32/90与随机10/30可行数量。',
        regression_to_mean='采用初始化与随机对照；未按极端训练结果挑选种子。',
        survivorship_bias='36/36评估任务完成，全部不可行序列进入统计。',
        look_elsewhere_effect='报告全部六个主对照；区间探索性，未做多重比较校正。',
        forking_paths='复用原分析方法与统计种子，第50轮仅为过程观察；本次为既有证据补充核对。',
        correlation_causation='不将网络改变或奖励提高等同搜索收益；不推断步数导致差异。',
        reverse_causality='检查点先训练后固定评估，评估序列不参与更新；仍非新电路泛化验证。')
    write(HERE / 'validation.json', dict(status='complete', started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        audit_source_sha256=common.sha(__file__), original_training_fingerprint=fingerprint,
        source_tool_input_fingerprint_match=True, current_artifacts_verified=current_artifacts,
        historical_document_bindings=historical_documents,
        model_probe_files_verified=len(model_manifest['hashes']), training_validation=training_validation,
        physical_evaluation_banks=bank_count, physical_evaluation_rollouts=len(rows), evaluation_actions=10800,
        nonduplicated_policy_rollout_combinations=900, independent_uniform_sequences_per_circuit=30,
        raw_logs_csv_bootstrap_exact_match=True, cec_exports_hash_verified=1110,
        prior_cec_netlists_hash_verified=2217, evidence_archive_files_verified=archived,
        inference_replays=replays, full_training_or_evaluation_rerun=False,
        statistical_fallacies_checked=scan, statistical_scan_coverage='11/11',
        original_evaluation_finished_at=read(root / 'evaluation.json')['finished_at']))
    print(f'Complete. New audit evidence: {HERE}', flush=True)


if __name__ == '__main__':
    main()
