"""Reconstruct committed evidence, five-step paired estimates, and equal-budget history."""
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

from common import (HERE, ROOT, GROUPS, POLICIES, EVAL_SEEDS, SNAPSHOTS, dump, load, now, returns_for, sha, state_hash, MAIN_POLICY, EVAL_PREFIXES, BUDGETS, BOOTSTRAP,
                    EXPECTED_TRAINING_RUNS, EXPECTED_PHYSICAL_BANKS, EXPECTED_PHYSICAL_ROLLOUTS,
                    EXPECTED_LOGICAL_ROLLOUTS, EPISODES, ITERATIONS, ASSESSMENT)
from drills.fpga_session import FPGASession

OLD = ROOT / 'experiments/learning-effectiveness'


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        if not keys:
            return
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def rank(row):
    return FPGASession.rank(row)


def hierarchical_bootstrap(differences, repetitions=BOOTSTRAP['repetitions'], seed=BOOTSTRAP['seed']):
    """Resample training rows, then common paired columns jointly (shared uniform bank)."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or not values.size:
        raise ValueError('Bootstrap needs a complete finite paired matrix.')
    if np.ptp(values) == 0:
        return [float(values.flat[0]), float(values.flat[0])]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        training = rng.integers(0, values.shape[0], size=values.shape[0])
        evaluation = rng.integers(0, values.shape[1], size=(1, values.shape[1]))
        estimates[i] = values[training[:, None], evaluation].mean()
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def policy_comparison(rows, name, baseline, seeds, prefix):
    matrices, feasible_matrices, per_seed, scores = [], [], [], []
    for seed in seeds:
        def bank(policy):
            return sorted((r for r in rows if r['policy'] == policy and r['circuit'] == name
                           and r['training_seed'] == seed and r['steps'] == prefix),
                          key=lambda r: r['evaluation_seed'])
        left, right = bank(baseline), bank(MAIN_POLICY)
        complete = len(left) == len(right) == len(EVAL_SEEDS) and all(
            r['status'] == 'complete' and r.get('task_status', r['status']) == 'complete' for r in left + right)
        if not complete:
            per_seed.append(dict(seed=seed, complete=False, pairs=sum(a['status'] == b['status'] == 'complete'
                for a, b in zip(left, right)), luts_reduction=None, percent=None,
                baseline_feasible=None, trained_feasible=None, feasibility_gain_pp=None,
                wins=None, ties=None, losses=None))
            continue
        if any([r['evaluation_seed'] for r in bank] != EVAL_SEEDS for bank in (left, right)):
            raise ValueError('Paired evaluation seeds are misaligned.')
        feasible = [100 * (int(b['best_feasible']) - int(a['best_feasible'])) for a, b in zip(left, right)]
        feasible_matrices.append(feasible)
        differences = None
        if all(r['best_feasible'] for r in left + right):
            differences = [a['best_luts'] - b['best_luts'] for a, b in zip(left, right)]
            matrices.append(differences)
        seed_scores = []
        for a, b in zip(left, right):
            ar = rank(dict(feasible=a['best_feasible'], luts=a['best_luts'], levels=a['best_levels']))
            br = rank(dict(feasible=b['best_feasible'], luts=b['best_luts'], levels=b['best_levels']))
            seed_scores.append(int(br < ar) - int(br > ar))
        scores.extend(seed_scores)
        per_seed.append(dict(seed=seed, complete=True, pairs=len(left),
            luts_reduction=float(np.mean(differences)) if differences is not None else None,
            percent=100 * float(np.mean(differences)) / float(np.mean([r['best_luts'] for r in left]))
                    if differences is not None else None,
            baseline_feasible=sum(r['best_feasible'] for r in left),
            trained_feasible=sum(r['best_feasible'] for r in right),
            feasibility_gain_pp=float(np.mean(feasible)), wins=sum(v > 0 for v in seed_scores),
            ties=sum(v == 0 for v in seed_scores), losses=sum(v < 0 for v in seed_scores)))
    complete = len(feasible_matrices) == len(seeds)
    all_feasible = complete and len(matrices) == len(seeds)
    return dict(circuit=name, steps=prefix, baseline=baseline, treatment=MAIN_POLICY,
        mean_luts_reduction=float(np.mean(matrices)) if all_feasible else None,
        ci95=hierarchical_bootstrap(matrices) if all_feasible else None,
        feasibility_gain_pp=float(np.mean(feasible_matrices)) if complete else None,
        feasibility_ci95=hierarchical_bootstrap(feasible_matrices) if complete else None,
        baseline_feasible=sum(p['baseline_feasible'] or 0 for p in per_seed),
        trained_feasible=sum(p['trained_feasible'] or 0 for p in per_seed), per_seed=per_seed,
        wins=sum(v > 0 for v in scores) if complete else None,
        ties=sum(v == 0 for v in scores) if complete else None,
        losses=sum(v < 0 for v in scores) if complete else None,
        complete=complete, complete_feasible=all_feasible)


def validate_log(path, cfg, circuit):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != cfg['protocol']['iterations'] + 1:
        raise ValueError(f'Unexpected step count: {path}')
    reward_game = FPGASession.__new__(FPGASession)
    reward_game.method, reward_game.circuit = cfg['method'], circuit
    previous, processed, sequence = None, [], list(cfg['protocol']['initial_sequence'])
    for i, row in enumerate(rows):
        if int(row['iteration']) != i:
            raise ValueError(f'Noncontiguous log: {path}')
        luts, levels, reward = int(row['luts']), int(row['levels']), float(row['reward'])
        if previous is not None:
            if row['optimization'] not in cfg['protocol']['actions']:
                raise ValueError('Action outside protocol.')
            reward_game.luts, reward_game.levels = previous['luts'], previous['levels']
            expected = reward_game._get_reward(luts, levels)
            sequence.append(row['optimization'])
        else:
            expected = 0
        if reward != expected:
            raise ValueError(f'Reward mismatch: {path}, step {i}')
        item = dict(iteration=i, optimization=row['optimization'], luts=luts, levels=levels,
                    reward=reward, feasible=levels <= circuit['max_levels'], sequence=list(sequence))
        processed.append(item)
        previous = item
    return processed


def assert_fields(actual, expected, keys, description):
    if any(actual.get(k) != expected.get(k) for k in keys):
        raise ValueError(description)


def training_tables(cfg):
    root = Path(cfg['runtime']['output_dir'])
    runs, curves, episodes, diagnostics, steps, validations, behavior = [], [], [], [], [], [], []
    for group in GROUPS:
        for name, circuit in cfg['protocol']['circuits'].items():
            for seed in cfg['protocol']['seeds']:
                folder = root / 'training' / group / name / f'seed-{seed}'
                status_record = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
                result = read(folder / 'result.json') if (folder / 'result.json').exists() else dict(
                    status='failed' if status_record.get('status') == 'failed' else 'missing',
                    episodes_completed=0, best=None, error=status_record.get('error'))
                checkpoint = load(folder / 'checkpoint.pt') if (folder / 'checkpoint.pt').exists() else None
                if checkpoint is not None:
                    if result['status'] == 'complete':
                        assert_fields(checkpoint, result, ('episodes_completed', 'best', 'rewards'),
                                      f'Result/checkpoint mismatch: {folder}')
                    else:
                        result.update(status='failed' if result['status'] == 'failed' else 'incomplete',
                                      episodes_completed=checkpoint['episodes_completed'], best=checkpoint['best'],
                                      rewards=checkpoint.get('rewards', []))
                count = result.get('episodes_completed', 0)
                if result['status'] == 'complete' and count != cfg['protocol']['episodes']:
                    raise ValueError(f'Complete task has incorrect budget: {folder}')
                diags = [d for d in read_jsonl(folder / 'diagnostics.jsonl') if d['episode'] <= count]
                all_steps = [s for s in read_jsonl(folder / 'steps.jsonl') if s['episode'] <= count]
                if len(diags) != count or len(all_steps) != count * cfg['protocol']['iterations']:
                    raise ValueError(f'Committed diagnostic/step coverage mismatch: {folder}')
                by_episode = {e: [s for s in all_steps if s['episode'] == e] for e in range(1, count + 1)}
                best, candidate, step_feasible, episode_feasible, terminal_feasible = None, 0, 0, 0, 0
                roundtrips, positive_roundtrips, raw_inputs, normalized_inputs = 0, 0, [], []
                for ep in range(1, count + 1):
                    logged = validate_log(folder / f'episodes/{ep}/log.csv', cfg, circuit)
                    rewards = [r['reward'] for r in logged[1:]]
                    raw_returns, processed = returns_for(rewards, cfg['method'])
                    diag = diags[ep - 1]
                    if diag['episode'] != ep or diag['reward_sum'] != sum(rewards):
                        raise ValueError(f'Episode diagnostic mismatch: {folder}')
                    for key, value in [('raw_returns_std', raw_returns.std()), ('processed_returns_std', processed.std())]:
                        if not np.isclose(diag[key], value, atol=1e-7):
                            raise ValueError(f'Return scale mismatch: {folder}')
                    if result.get('rewards') and result['rewards'][ep - 1] != sum(rewards):
                        raise ValueError(f'Checkpoint rewards mismatch: {folder}')
                    for i, row in enumerate(logged):
                        if i == 0 and not cfg['protocol']['evaluation']['include_initial']:
                            continue
                        ranked = dict(**row, episode=ep)
                        if best is None or rank(ranked) < rank(best):
                            best = ranked
                        if i:
                            evidence = by_episode[ep][i - 1]
                            assert_fields(evidence, row, ('iteration', 'luts', 'levels', 'reward', 'optimization'),
                                          f'Instrumented evidence differs from ABC log: {folder}')
                            if cfg['protocol']['actions'][evidence['action']] != row['optimization']:
                                raise ValueError(f'Action index mismatch: {folder}')
                            candidate += 1
                            step_feasible += int(row['feasible'])
                            curves.append(dict(group=group, circuit=name, seed=seed, candidate=candidate,
                                luts=best['luts'], levels=best['levels'], feasible=best['feasible']))
                    eligible = logged if cfg['protocol']['evaluation']['include_initial'] else logged[1:]
                    episode_best = min(eligible, key=rank)
                    episode_feasible += int(episode_best['feasible'])
                    terminal_feasible += int(logged[-1]['feasible'])
                    episodes.append(dict(group=group, circuit=name, seed=seed, episode=ep,
                        best_luts=episode_best['luts'], best_levels=episode_best['levels'],
                        best_feasible=episode_best['feasible'], terminal_luts=logged[-1]['luts'],
                        terminal_levels=logged[-1]['levels'], terminal_feasible=logged[-1]['feasible'], reward=sum(rewards)))
                    diagnostics.append(dict(group=group, circuit=name, seed=seed, **diag))
                    for i in range(2, len(logged)):
                        a, b, c = logged[i - 2:i + 1]
                        if a['luts'] == c['luts'] != b['luts'] and a['feasible'] and b['feasible'] and c['feasible']:
                            roundtrips += 1
                            positive_roundtrips += int(b['reward'] + c['reward'] > 0)
                    raw_inputs.extend(s['raw_state'] for s in by_episode[ep])
                    normalized_inputs.extend(s['normalized_state'] for s in by_episode[ep])
                    if not np.equal(by_episode[ep][0]['normalized_state'], 0).all():
                        raise ValueError(f'First normalized state is not zero: {folder}')
                if best is not None:
                    assert_fields(best, result.get('best') or {},
                                  ('luts', 'levels', 'feasible', 'sequence', 'episode', 'iteration'),
                                  f'Best does not replay from committed logs: {folder}')
                if result['status'] == 'complete':
                    if checkpoint is None:
                        raise ValueError(f'Complete training lacks checkpoint: {folder}')
                    if 'Networks are equivalent' not in (folder / 'equivalence.log').read_text():
                        raise ValueError(f'Missing training CEC: {folder}')
                    for snapshot in SNAPSHOTS:
                        saved = load(folder / f'snapshots/{snapshot}.pt')
                        if saved['episodes_completed'] != snapshot:
                            raise ValueError(f'Snapshot episode mismatch: {folder}')
                        if group != 'trained' and state_hash(saved['network']) != state_hash(load(folder / 'snapshots/0.pt')['network']):
                            raise ValueError(f'Frozen snapshot changed: {folder}')
                best_candidate = (best['episode'] - 1) * cfg['protocol']['iterations'] + best['iteration'] if best else None
                runs.append(dict(group=group, circuit=name, seed=seed, status=result['status'],
                    episodes_completed=count, training_seconds=result.get('training_seconds'), error=result.get('error'),
                    max_levels=circuit['max_levels'], best=best, best_luts=best['luts'] if best else None,
                    best_levels=best['levels'] if best else None, best_feasible=best['feasible'] if best else None,
                    best_episode=best['episode'] if best else None, best_iteration=best['iteration'] if best else None,
                    best_candidate=best_candidate, candidates=candidate, feasible_candidates=step_feasible,
                    feasible_candidate_rate=step_feasible / candidate if candidate else None,
                    feasible_episodes=episode_feasible, feasible_episode_rate=episode_feasible / count if count else None,
                    feasible_terminals=terminal_feasible, feasible_terminal_rate=terminal_feasible / count if count else None))
                steps.extend(dict(group=group, circuit=name, seed=seed, **s) for s in all_steps)
                validations.append(dict(group=group, circuit=name, seed=seed, status=result['status'],
                    completed_episodes=count, replayed_steps=candidate, best_replayed=best is not None,
                    checkpoint_checked=checkpoint is not None))
                if raw_inputs:
                    behavior.append(dict(group=group, circuit=name, seed=seed, positive_lut_roundtrips=positive_roundtrips,
                        lut_roundtrips=roundtrips, zero_reward_fraction=float(np.mean([s['reward'] == 0 for s in all_steps])),
                        raw_constant_features=[f for f, v in zip(cfg['method']['features'], np.ptp(raw_inputs, axis=0)) if v == 0],
                        normalized_zero_features=[f for f, v in zip(cfg['method']['features'], np.max(np.abs(normalized_inputs), axis=0)) if v == 0]))
    return runs, curves, episodes, diagnostics, steps, validations, behavior


def evaluation_tables(cfg):
    root = Path(cfg['runtime']['output_dir'])
    rows, cases, physical_checks = [], [], []
    cache = {}
    for policy in POLICIES:
        for name, circuit in cfg['protocol']['circuits'].items():
            for seed in cfg['protocol']['seeds']:
                physical_seed = 0 if policy == 'uniform' else seed
                bank_id = f'{policy}/{name}/seed-{physical_seed}'
                folder = root / 'evaluation' / bank_id
                if bank_id not in cache:
                    status_record = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
                    if (folder / 'result.json').exists():
                        result = read(folder / 'result.json')
                    else:
                        partial = [read(p) for p in sorted(folder.glob('rollout-*/rollout.json'))]
                        result = dict(status='failed' if status_record.get('status') == 'failed' else
                                      ('incomplete' if partial else 'missing'), rollouts=partial, error=status_record.get('error'))
                    records = result.get('rollouts', [])
                    actual_seeds = [r['evaluation_seed'] for r in records]
                    if len(set(actual_seeds)) != len(actual_seeds) or any(s not in EVAL_SEEDS for s in actual_seeds):
                        raise ValueError(f'Invalid/duplicate evaluation seeds: {folder}')
                    if result['status'] == 'complete' and sorted(actual_seeds) != EVAL_SEEDS:
                        raise ValueError(f'Complete bank lacks seed coverage: {folder}')
                    validated = {}
                    checkpoint_group = 'uniform' if policy == 'uniform' else 'trained'
                    checkpoint_episode = int(policy.split('-')[1]) if policy.startswith('trained-') else 0
                    checkpoint = root / 'training' / checkpoint_group / name / f'seed-{physical_seed}/snapshots/{checkpoint_episode}.pt'
                    for record in records:
                        destination = folder / f'rollout-{record["evaluation_seed"]}'
                        raw_record = read(destination / 'rollout.json')
                        if record != raw_record:
                            raise ValueError(f'Aggregate/rollout record mismatch: {destination}')
                        if not checkpoint.exists() or record.get('checkpoint_sha256') != sha(checkpoint):
                            raise ValueError(f'Evaluation checkpoint fingerprint mismatch: {destination}')
                        logged = validate_log(destination / 'episodes/1/log.csv', cfg, circuit)
                        if record['actions'] != [cfg['protocol']['actions'].index(r['optimization']) for r in logged[1:]]:
                            raise ValueError(f'Evaluation actions mismatch: {destination}')
                        if record['rewards'] != [r['reward'] for r in logged[1:]]:
                            raise ValueError(f'Evaluation rewards mismatch: {destination}')
                        assert_fields(record['terminal'], logged[-1], ('luts', 'levels', 'feasible'),
                                      f'Evaluation terminal mismatch: {destination}')
                        eligible = logged if cfg['protocol']['evaluation']['include_initial'] else logged[1:]
                        best = min(eligible, key=rank)
                        assert_fields(record['best'], best, ('luts', 'levels', 'feasible', 'sequence', 'iteration'),
                                      f'Evaluation best mismatch: {destination}')
                        by_prefix = {r['steps']: r for r in record.get('prefixes', [])}
                        if set(by_prefix) != set(EVAL_PREFIXES):
                            raise ValueError(f'Evaluation prefix coverage mismatch: {destination}')
                        verified_prefixes = {}
                        for prefix in EVAL_PREFIXES:
                            subset = logged[:prefix + 1] if cfg['protocol']['evaluation']['include_initial'] else logged[1:prefix + 1]
                            expected = dict(best=min(subset, key=rank), terminal=logged[prefix],
                                            reward=sum(r['reward'] for r in logged[1:prefix + 1]))
                            assert_fields(by_prefix[prefix]['best'], expected['best'],
                                ('luts', 'levels', 'feasible', 'sequence', 'iteration'), f'Prefix best mismatch: {destination}')
                            assert_fields(by_prefix[prefix]['terminal'], expected['terminal'],
                                          ('luts', 'levels', 'feasible'), f'Prefix terminal mismatch: {destination}')
                            if by_prefix[prefix]['reward'] != expected['reward']:
                                raise ValueError(f'Prefix reward mismatch: {destination}')
                            verified_prefixes[prefix] = expected
                        if 'Networks are equivalent' not in (destination / 'equivalence.log').read_text():
                            raise ValueError(f'Missing evaluation CEC: {destination}')
                        validated[record['evaluation_seed']] = (record, verified_prefixes)
                    physical_checks.append(dict(physical_bank_id=bank_id, status=result['status'],
                        completed=len(validated), expected=len(EVAL_SEEDS), error=result.get('error')))
                    cache[bank_id] = result, validated
                result, validated = cache[bank_id]
                for prefix in EVAL_PREFIXES:
                    prefix_rows = []
                    for evaluation_seed in EVAL_SEEDS:
                        base = dict(policy=policy, circuit=name, training_seed=seed, evaluation_seed=evaluation_seed,
                                    steps=prefix, physical_bank_id=bank_id, shared_bank=policy == 'uniform',
                                    physical_training_seed=physical_seed, task_status=result['status'], error=result.get('error'))
                        if evaluation_seed not in validated:
                            row = dict(**base, status='missing', best_luts=None, best_levels=None, best_feasible=None,
                                       terminal_luts=None, terminal_levels=None, terminal_feasible=None, reward=None)
                        else:
                            record, prefixes = validated[evaluation_seed]
                            value = prefixes[prefix]
                            row = dict(**base, status=record.get('status', 'complete'),
                                checkpoint_sha256=record['checkpoint_sha256'], best_luts=value['best']['luts'],
                                best_levels=value['best']['levels'], best_feasible=value['best']['feasible'],
                                terminal_luts=value['terminal']['luts'], terminal_levels=value['terminal']['levels'],
                                terminal_feasible=value['terminal']['feasible'], reward=value['reward'])
                        rows.append(row)
                        prefix_rows.append(row)
                    complete = result['status'] == 'complete' and all(r['status'] == 'complete' for r in prefix_rows)
                    best_all = complete and all(r['best_feasible'] for r in prefix_rows)
                    terminal_all = complete and all(r['terminal_feasible'] for r in prefix_rows)
                    cases.append(dict(policy=policy, circuit=name, training_seed=seed, steps=prefix,
                        status=result['status'], completed=sum(r['status'] == 'complete' for r in prefix_rows),
                        feasible=sum(bool(r['best_feasible']) for r in prefix_rows),
                        best_luts_mean=float(np.mean([r['best_luts'] for r in prefix_rows])) if best_all else None,
                        terminal_luts_mean=float(np.mean([r['terminal_luts'] for r in prefix_rows])) if terminal_all else None,
                        terminal_feasible=sum(bool(r['terminal_feasible']) for r in prefix_rows),
                        terminal_levels_mean=float(np.mean([r['terminal_levels'] for r in prefix_rows])) if complete else None,
                        reward_mean=float(np.mean([r['reward'] for r in prefix_rows])) if complete else None,
                        physical_bank_id=bank_id, shared_bank=policy == 'uniform', error=result.get('error')))
    return rows, cases, physical_checks


def budget_and_hits(cfg, curves, runs):
    """Optional, read-only history; current execution never needs historical weights."""
    path = OLD / 'trajectories.csv'
    records = []
    if path.exists():
        with path.open() as stream:
            records = list(csv.DictReader(stream))
    budget = cfg['protocol']['episodes'] * cfg['protocol']['iterations']
    rows = []
    for group in GROUPS:
        for name in cfg['protocol']['circuits']:
            for seed in cfg['protocol']['seeds']:
                value = next((r for r in records if r['group'] == group and r['circuit'] == name
                    and int(r['seed']) == seed and int(r['candidate']) == budget), None)
                rows.append(dict(experiment='100x10', group=group, circuit=name, seed=seed,
                    budget=budget, status='complete' if value else 'missing',
                    best_luts=int(value['luts']) if value else None,
                    best_levels=int(value['levels']) if value else None,
                    best_feasible=value['feasible'] == 'True' if value else None))
    return rows, [], {str(path.relative_to(ROOT)): sha(path)} if path.exists() else {}


def search_comparisons(runs, cfg):
    rows = []
    for name in cfg['protocol']['circuits']:
        for baseline in ('frozen', 'uniform'):
            pairs = []
            for seed in cfg['protocol']['seeds']:
                a = next(r for r in runs if r['circuit'] == name and r['seed'] == seed and r['group'] == baseline)
                b = next(r for r in runs if r['circuit'] == name and r['seed'] == seed and r['group'] == 'trained')
                complete = a['status'] == b['status'] == 'complete' and a['best'] is not None and b['best'] is not None
                feasible = complete and a['best_feasible'] and b['best_feasible']
                difference = a['best_luts'] - b['best_luts'] if feasible else None
                score = int(rank(b['best']) < rank(a['best'])) - int(rank(b['best']) > rank(a['best'])) if complete else None
                pairs.append(dict(seed=seed, complete=complete, difference=difference,
                    percent=100 * difference / a['best_luts'] if feasible else None, score=score))
            all_feasible = all(p['difference'] is not None for p in pairs)
            differences = [p['difference'] for p in pairs]
            mean = float(np.mean(differences)) if all_feasible else None
            ref = [r['best_luts'] for r in runs if r['circuit'] == name and r['group'] == baseline]
            rows.append(dict(circuit=name, baseline=baseline, differences=differences, per_seed=pairs,
                mean_luts_reduction=mean, percent=100 * mean / float(np.mean(ref)) if mean is not None else None,
                wins=sum(p['score'] == 1 for p in pairs), ties=sum(p['score'] == 0 for p in pairs),
                losses=sum(p['score'] == -1 for p in pairs), complete=all(p['complete'] for p in pairs)))
    return rows


def generate(cfg):
    runs, curves, episodes, diagnostics, steps, validation, behavior = training_tables(cfg)
    evaluation_rows, evaluation_cases, physical_checks = evaluation_tables(cfg)
    comparisons = [policy_comparison(evaluation_rows, name, baseline, cfg['protocol']['seeds'], prefix)
                   for prefix in EVAL_PREFIXES for name in cfg['protocol']['circuits'] for baseline in ('initial', 'uniform')]
    budgets, hits, old_sources = budget_and_hits(cfg, curves, runs)
    new_budgets = []
    for group in GROUPS:
        for name in cfg['protocol']['circuits']:
            for seed in cfg['protocol']['seeds']:
                by_candidate = {r['candidate']: r for r in curves if
                    (r['group'], r['circuit'], r['seed']) == (group, name, seed)}
                for budget in BUDGETS:
                    value = by_candidate.get(budget)
                    new_budgets.append(dict(experiment='200x5', group=group, circuit=name, seed=seed,
                        budget=budget, status='complete' if value else 'missing',
                        best_luts=value['luts'] if value else None, best_levels=value['levels'] if value else None,
                        best_feasible=value['feasible'] if value else None))
    root = Path(cfg['runtime']['output_dir'])
    processes = {}
    for phase in ('training', 'evaluation'):
        path = root / 'logs' / phase / 'processes.json'
        processes[phase] = read(path) if path.exists() else []
    failures = [dict(phase=phase, **row) for phase, records in processes.items()
                for row in records if row['exit_code'] != 0]
    for run in runs:
        label = f'{run["group"]}-{run["circuit"]}-{run["seed"]}'
        failure = next((r for r in failures if r['phase'] == 'training' and r['label'] == label), None)
        if failure:
            run['status'] = 'timed_out' if failure['timed_out'] else 'failed'
            run['error'] = '30分钟硬超时' if failure['timed_out'] else f'退出码 {failure["exit_code"]}'
            next(r for r in validation if (r['group'],r['circuit'],r['seed']) ==
                 (run['group'],run['circuit'],run['seed']))['status'] = run['status']
    latest_diagnostics = [max(bank, key=lambda r: r['episode']) for run in runs
        if (bank := [r for r in diagnostics if (r['group'],r['circuit'],r['seed']) ==
                     (run['group'],run['circuit'],run['seed'])])]
    evaluation_manifest = read(root / 'evaluation.json') if (root / 'evaluation.json').exists() else {}
    summary = dict(protocol=dict(episodes=cfg['protocol']['episodes'], iterations=cfg['protocol']['iterations'],
            seeds=cfg['protocol']['seeds'], circuits=cfg['protocol']['circuits'], snapshots=list(SNAPSHOTS),
            evaluation_prefixes=list(EVAL_PREFIXES), assessment=ASSESSMENT, groups=list(GROUPS), policies=list(POLICIES),
            expected_training_runs=EXPECTED_TRAINING_RUNS, expected_physical_banks=EXPECTED_PHYSICAL_BANKS,
            expected_physical_rollouts=EXPECTED_PHYSICAL_ROLLOUTS, expected_logical_rollouts=EXPECTED_LOGICAL_ROLLOUTS), training=runs, evaluation=evaluation_cases,
        comparisons=comparisons, search_comparisons=search_comparisons(runs, cfg), behavior=behavior,
        old_budget=budgets, new_budget=new_budgets, old_sources=old_sources,
        diagnostic_summary=latest_diagnostics, failures=failures,
        skipped_evaluation=evaluation_manifest.get('skipped', []),
        training_complete=sum(r['status'] == 'complete' for r in runs),
        evaluation_complete=sum(r['status'] == 'complete' and r['steps'] == ITERATIONS for r in evaluation_cases),
        physical_banks=physical_checks, physical_banks_complete=sum(r['status'] == 'complete' for r in physical_checks),
        physical_rollouts=sum(r['completed'] for r in physical_checks),
        logical_rollouts=sum(r['status'] == 'complete' and r['steps'] == ITERATIONS for r in evaluation_rows),
        baseline=read(root / 'baseline.json') if (root / 'baseline.json').exists() else {},
        netlist_validation=read(HERE / 'netlist-validation.json') if (HERE / 'netlist-validation.json').exists() else {},
        validation=read(HERE / 'validation.json') if (HERE / 'validation.json').exists() else {},
        execution={})
    for phase, filename in [('training', 'provenance.json'), ('evaluation', 'evaluation-provenance.json')]:
        if (HERE / filename).exists():
            record = read(HERE / filename)
            summary['execution'][phase] = {k: record.get(k) for k in ('started_at', 'finished_at', 'elapsed_seconds', 'status')}
            if record.get('started_at') and record.get('finished_at'):
                summary['execution'][phase]['elapsed_seconds'] = (
                    datetime.fromisoformat(record['finished_at']) - datetime.fromisoformat(record['started_at'])).total_seconds()
    dump(HERE / 'summary.json', summary)
    dump(HERE / 'log-validation.json', dict(training=validation, evaluation=physical_checks,
        evidence_replayed=True,
        committed_training_steps=len(steps), physical_evaluation_rollouts=summary['physical_rollouts'],
        logical_evaluation_prefix_rows=len(evaluation_rows), status='complete' if
        summary['training_complete'] == EXPECTED_TRAINING_RUNS and summary['physical_banks_complete'] == EXPECTED_PHYSICAL_BANKS else 'incomplete'))
    training_csv = [{k: v for k, v in r.items() if k != 'best'} for r in runs]
    for filename, records in [('training.csv', training_csv), ('trajectories.csv', curves), ('episodes.csv', episodes),
                             ('diagnostics.csv', diagnostics), ('steps.csv', steps), ('evaluation.csv', evaluation_rows),
                             ('comparisons.csv', comparisons), ('old-budget.csv', budgets), ('search-budget.csv', new_budgets)]:
        write_csv(HERE / filename, records)
    from report import write_report
    write_report(summary, HERE)
    dump(HERE / 'analysis-provenance.json', dict(command=[sys.executable, *sys.argv], finished_at=now(),
        sources={str(p.relative_to(ROOT)): sha(p) for p in (HERE / 'analysis.py', HERE / 'evaluate.py', HERE / 'protocol.yml') if p.exists()},
        old_sources=old_sources, bootstrap_repetitions=BOOTSTRAP['repetitions'], bootstrap_seed=BOOTSTRAP['seed'],
        pairing='training rows first; common paired evaluation columns jointly; uniform bank shared across rows',
        independent_training_seeds=cfg['protocol']['seeds'], evaluation_seeds=EVAL_SEEDS,
        physical_expected_banks=EXPECTED_PHYSICAL_BANKS, physical_expected_rollouts=EXPECTED_PHYSICAL_ROLLOUTS,
        logical_expected_rollouts_per_prefix=EXPECTED_LOGICAL_ROLLOUTS,
        evaluation_prefixes=list(EVAL_PREFIXES)))
    print(f'Report: {HERE / "report.md"}; training {summary["training_complete"]}/{EXPECTED_TRAINING_RUNS}; '
          f'physical evaluation {summary["physical_rollouts"]}/{EXPECTED_PHYSICAL_ROLLOUTS}', flush=True)
    return summary

