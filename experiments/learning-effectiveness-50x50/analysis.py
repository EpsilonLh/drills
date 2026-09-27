"""Reconstruct committed evidence, paired prefix estimates, and both budget analyses."""
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

from common import HERE, ROOT, GROUPS, POLICIES, EVAL_SEEDS, SNAPSHOTS, dump, load, now, returns_for, sha, state_hash
from drills.fpga_session import FPGASession

EVAL_PREFIXES = (5, 10, 20, 30, 40, 50)
OLD_BUDGETS = (50, 100, 250, 500, 750, 1000)
TARGETS = {'resyn2': {'int2float': 48, 'i2c': 322, 'max': 777},
           'old_best': {'int2float': 43, 'i2c': 300, 'max': 763}}
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


def hierarchical_bootstrap(differences, repetitions=10000, seed=20260927):
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
        left, right = bank(baseline), bank('trained-50')
        complete = len(left) == len(right) == len(EVAL_SEEDS) and all(
            r['status'] == 'complete' and r.get('task_status', r['status']) == 'complete' for r in left + right)
        if not complete:
            per_seed.append(dict(seed=seed, complete=False, pairs=sum(a['status'] == b['status'] == 'complete'
                for a, b in zip(left, right)), luts_reduction=None, percent=None,
                baseline_feasible=None, trained_feasible=None, feasibility_gain_pp=None,
                wins=None, ties=None, losses=None))
            continue
        if [r['evaluation_seed'] for r in left] != [r['evaluation_seed'] for r in right]:
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
    return dict(circuit=name, steps=prefix, baseline=baseline, treatment='trained-50',
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
    old_path = OLD / 'trajectories.csv'
    old_curves = []
    with old_path.open() as stream:
        old_records = list(csv.DictReader(stream))
    for row in old_records:
        old_curves.append(dict(group=row['group'], circuit=row['circuit'], seed=int(row['seed']),
            candidate=int(row['candidate']), luts=int(row['luts']), levels=int(row['levels']), feasible=row['feasible'] == 'True'))
    budget_rows, hits = [], []
    old_groups = {(g, n, s): [r for r in old_curves if (r['group'], r['circuit'], r['seed']) == (g, n, s)]
                  for g in GROUPS for n in cfg['protocol']['circuits'] for s in cfg['protocol']['seeds']}
    for (group, name, seed), values in old_groups.items():
        for budget in OLD_BUDGETS:
            value = next((r for r in values if r['candidate'] == budget), None)
            budget_rows.append(dict(experiment='100x10', group=group, circuit=name, seed=seed,
                budget=budget, status='complete' if value else 'missing', best_luts=value['luts'] if value else None,
                best_levels=value['levels'] if value else None, best_feasible=value['feasible'] if value else None))
    for label, values, expected in [('100x10', old_curves, 1000), ('50x50', curves, cfg['protocol']['episodes'] * cfg['protocol']['iterations'])]:
        for group in GROUPS:
            for name in cfg['protocol']['circuits']:
                for seed in cfg['protocol']['seeds']:
                    records = [r for r in values if (r['group'], r['circuit'], r['seed']) == (group, name, seed)]
                    records.sort(key=lambda r: r['candidate'])
                    observed = records[-1]['candidate'] if records else 0
                    run = next((r for r in runs if (r['group'], r['circuit'], r['seed']) == (group, name, seed)), {})
                    status = 'complete' if observed == expected else run.get('status', 'missing')
                    for target, thresholds in TARGETS.items():
                        first = next((r for r in records if r['feasible'] and r['luts'] <= thresholds[name]), None)
                        hits.append(dict(experiment=label, group=group, circuit=name, seed=seed,
                            target=target, target_luts=thresholds[name], max_levels=cfg['protocol']['circuits'][name]['max_levels'],
                            budget=expected, observed_candidates=observed, status=status,
                            hit=first is not None, first_candidate=first['candidate'] if first else None,
                            censored=first is None, censoring='预算内未达到' if first is None and observed == expected
                                else ('未完成预算' if first is None else ''),
                            first_episode=(first['candidate'] - 1) // (10 if label == '100x10' else cfg['protocol']['iterations']) + 1 if first else None,
                            first_iteration=(first['candidate'] - 1) % (10 if label == '100x10' else cfg['protocol']['iterations']) + 1 if first else None))
    return budget_rows, hits, {str(old_path.relative_to(ROOT)): sha(old_path)}


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
                for budget in (50, 100, 250, 500, 750, 1000, 1500, 2000, 2500):
                    value = by_candidate.get(budget)
                    new_budgets.append(dict(experiment='50x50', group=group, circuit=name, seed=seed,
                        budget=budget, status='complete' if value else 'missing',
                        best_luts=value['luts'] if value else None, best_levels=value['levels'] if value else None,
                        best_feasible=value['feasible'] if value else None))
    root = Path(cfg['runtime']['output_dir'])
    summary = dict(protocol=dict(episodes=cfg['protocol']['episodes'], iterations=cfg['protocol']['iterations'],
            seeds=cfg['protocol']['seeds'], circuits=cfg['protocol']['circuits'], snapshots=list(SNAPSHOTS),
            evaluation_prefixes=list(EVAL_PREFIXES)), training=runs, evaluation=evaluation_cases,
        comparisons=comparisons, search_comparisons=search_comparisons(runs, cfg), behavior=behavior,
        old_budget=budgets, new_budget=new_budgets, target_hits=hits, old_sources=old_sources,
        diagnostic_summary=[r for r in diagnostics if r['episode'] == cfg['protocol']['episodes']],
        training_complete=sum(r['status'] == 'complete' for r in runs),
        evaluation_complete=sum(r['status'] == 'complete' and r['steps'] == 50 for r in evaluation_cases),
        physical_banks=physical_checks, physical_banks_complete=sum(r['status'] == 'complete' for r in physical_checks),
        physical_rollouts=sum(r['completed'] for r in physical_checks),
        logical_rollouts=sum(r['status'] == 'complete' and r['steps'] == 50 for r in evaluation_rows),
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
        committed_training_steps=len(steps), physical_evaluation_rollouts=summary['physical_rollouts'],
        logical_evaluation_prefix_rows=len(evaluation_rows), status='complete' if
        summary['training_complete'] == 27 and summary['physical_banks_complete'] == 30 else 'incomplete'))
    training_csv = [{k: v for k, v in r.items() if k != 'best'} for r in runs]
    for filename, records in [('training.csv', training_csv), ('trajectories.csv', curves), ('episodes.csv', episodes),
                             ('diagnostics.csv', diagnostics), ('steps.csv', steps), ('evaluation.csv', evaluation_rows),
                             ('comparisons.csv', comparisons), ('old-budget.csv', budgets), ('search-budget.csv', new_budgets),
                             ('target-hits.csv', hits)]:
        write_csv(HERE / filename, records)
    write_report(summary)
    dump(HERE / 'analysis-provenance.json', dict(command=[sys.executable, *sys.argv], finished_at=now(),
        sources={str(p.relative_to(ROOT)): sha(p) for p in (HERE / 'analysis.py', HERE / 'evaluate.py', HERE / 'protocol.yml') if p.exists()},
        old_sources=old_sources, bootstrap_repetitions=10000, bootstrap_seed=20260927,
        pairing='training rows first; common paired evaluation columns jointly; uniform bank shared across rows',
        independent_training_seeds=cfg['protocol']['seeds'], evaluation_seeds=EVAL_SEEDS,
        physical_expected_banks=30, physical_expected_rollouts=900, logical_expected_rollouts_per_prefix=1080,
        evaluation_prefixes=list(EVAL_PREFIXES)))
    print(f'Report: {HERE / "report.md"}; training {summary["training_complete"]}/27; '
          f'physical evaluation {summary["physical_rollouts"]}/900', flush=True)
    return summary


def format_value(value, digits=3):
    return '未汇总' if value is None else f'{value:.{digits}f}'


def interval(value):
    return '未汇总' if value is None else '[' + ', '.join(f'{v:.3f}' for v in value) + ']'


def count_rate(count, total):
    return f'{count}/{total} ({count / total:.1%})' if total else '0/0（未完成）'


def conclusion_lines(summary):
    trained = [r for r in summary['diagnostic_summary'] if r['group'] == 'trained']
    fixed = [r for r in summary['diagnostic_summary'] if r['group'] != 'trained']
    changed = sum(r['actor_drift_rms'] > 0 and r['critic_drift_rms'] > 0 for r in trained)
    policies = sum(r['policy_kl'] > 0 for r in trained)
    unchanged = sum(r['actor_drift_rms'] == 0 and r['critic_drift_rms'] == 0 for r in fixed)
    lines = [f'权重更新：训练组 {changed}/9 次 Actor/Critic 均改变；固定探针动作分布 {policies}/9 次改变；'
             f'冻结及 uniform {unchanged}/18 次参数保持不变。完整终轮诊断才计入这些分母。']
    for name in summary['protocol']['circuits']:
        effects = []
        for budget in OLD_BUDGETS:
            a = [r for r in summary['old_budget'] if r['circuit'] == name and r['group'] == 'frozen' and r['budget'] == budget]
            b = [r for r in summary['old_budget'] if r['circuit'] == name and r['group'] == 'trained' and r['budget'] == budget]
            effects.append(float(np.mean([x['best_luts'] - y['best_luts'] for x, y in zip(a, b)]))
                           if len(a) == len(b) == 3 and all(r['best_feasible'] for r in a + b) else None)
        lines.append(f'旧{name}在50/100/250/500/750/1,000候选处，训练相对冻结平均少LUT为 '
                     + ' / '.join(format_value(v) for v in effects) + '。')
    lines.append('旧预算曲线未呈现三个电路一致的“训练早期领先、冻结随后追平”；'
                 '个别预算或相对uniform的优势不足以单独支持探索预算抹平学习优势这一解释。')
    for row in summary['search_comparisons']:
        if row['baseline'] == 'frozen':
            lines.append(f'{row["circuit"]}：2,500 候选搜索中，训练相对冻结平均少 '
                f'{format_value(row["mean_luts_reduction"])} LUT，逐种子差值 {row["differences"]}，'
                f'胜/平/负 {row["wins"]}/{row["ties"]}/{row["losses"]}。')
    if summary['physical_banks_complete'] != 30:
        return lines + ['独立评估尚未全部完成，暂不作最终学习收益判断。']
    main = [r for r in summary['comparisons'] if r['steps'] == 50]
    for row in main:
        if row['baseline'] != 'initial':
            continue
        if row['mean_luts_reduction'] is not None:
            lines.append(f'{row["circuit"]}：第50轮固定策略相对初始化/冻结平均减少 '
                f'{row["mean_luts_reduction"]:.3f} LUT，探索性95%区间 {interval(row["ci95"])}。')
        else:
            lines.append(f'{row["circuit"]}：第50轮固定策略可行 {row["trained_feasible"]}/90，初始化/冻结 '
                f'{row["baseline_feasible"]}/90；可行率变化 {format_value(row["feasibility_gain_pp"])} 个百分点，'
                f'探索性95%区间 {interval(row["feasibility_ci95"])}，未汇总筛选后的LUT均值。')
    consistent = all(r['complete'] and (r['feasibility_gain_pp'] > 0 or
        (r['feasibility_gain_pp'] == 0 and r['mean_luts_reduction'] is not None and r['mean_luts_reduction'] > 0)) for r in main)
    lines.append('完整50步评估对三个电路和两种基线的平均改善方向一致；仍须检查逐种子波动和探索性区间，不能据此宣称普遍有效。'
        if consistent else '完整50步评估未呈现对三个电路、两种基线均一致的收益；这不否定参数更新，也不证明策略等效或其他设置无效。')
    return lines


def write_report(summary):
    protocol = summary['protocol']
    episode_count, length = protocol['episodes'], protocol['iterations']
    lines = ['# DRiLLS 50轮 × 50步：学习收益与搜索预算报告', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run + validate',
        '- Origin Date: 2026-09-27', '- Verification Status: ' + ('VERIFIED（日志重算；另见测试及CEC验证记录）'
             if summary['training_complete'] == 27 and summary['physical_banks_complete'] == 30 else 'INCOMPLETE'),
        '- Version Label: learning_effectiveness_50x50_v1', '', '## 主要结论', '',
        f'训练搜索完成 {summary["training_complete"]}/27 次，物理独立评估完成 {summary["physical_rollouts"]}/900 条，'
        f'对应 {summary["logical_rollouts"]}/1,080 条逻辑配对行（每个前缀）。', '', *conclusion_lines(summary), '',
        '**50轮×50步同时增加搜索动作预算（1,000→2,500）、延长单条序列（10→50）、减少更新次数（100→50）。**'
        '回报标准化范围和求和损失长度也改变，因此跨设置的改善不能直接归因于学习。预算抹平假设由旧数据预算曲线单独检查；'
        '新实验回答长序列条件下固定训练策略是否优于初始化与随机策略。', '', '## 条件和统计口径', '',
        f'三组 trained/frozen/uniform、种子0/1/2，每次 {episode_count}轮×{length}步；'
        'int2float/i2c/max 的最大层数为3/4/41，LUT6。算法、奖励、episode_welford状态归一化、网络、Adam和学习率保持原设置。'
        'trained更新网络；frozen从初始化起保持Actor/Critic及Adam不变；uniform固定以1/7概率抽取七种动作。'
        'uniform训练种子控制随机动作，网络初始化不影响动作选择。', '',
        '保存第0/10/25/50轮模型；独立评估预定使用第0/25/50轮，评估种子10000–10029。'
        '各序列重新从原电路开始，全程停止学习，前缀5/10/20/30/40/50步均来自同一条50步轨迹。'
        '初始映射也可成为最佳解，但不计入动作候选数。第一轮和初始状态公平性见validation.json。', '',
        '物理评估有30个bank：3电路×3训练种子×3模型检查点，加3个uniform bank，共900条序列。'
        'uniform每个电路只执行一份30条基线；evaluation.csv为配对将其共享给三个训练种子，形成每前缀1,080行。'
        '共享行带physical_bank_id和shared_bank标记，不增加独立样本量。冻结通过不变性检查后共享初始化评估。', '',
        '主指标为可行率和最佳可行LUT。任何配对存在不可行时，不使用筛除失败后的LUT均值；'
        '改报完整可行率和可行优先胜/平/负。排序为可行优先，可行时先LUT再层数，不可行时先层数再LUT。'
        '失败、未完成和未命中目标均保留。终点层数、LUT和奖励为辅助指标。', '', '## 旧100×10数据：预算效应', '',
        '| 电路 | 预算 | trained逐种子最佳LUT | frozen逐种子最佳LUT | uniform逐种子最佳LUT | 训练相对冻结平均减少LUT | 训练相对uniform平均减少LUT |',
        '|---|---:|---|---|---|---:|---:|']
    for name in protocol['circuits']:
        for budget in OLD_BUDGETS:
            banks = [[r for r in summary['old_budget'] if r['circuit'] == name and r['budget'] == budget and r['group'] == g] for g in GROUPS]
            values = [' / '.join(str(r['best_luts']) if r['best_feasible'] else '不可行' for r in bank) for bank in banks]
            effects = []
            for ref in banks[1:]:
                valid = all(r['status'] == 'complete' and r['best_feasible'] for r in banks[0] + ref)
                effects.append(float(np.mean([a['best_luts'] - b['best_luts'] for a, b in zip(ref, banks[0])])) if valid else None)
            lines.append(f'| {name} | {budget} | ' + ' | '.join(values) + ' | ' + ' | '.join(format_value(v) for v in effects) + ' |')
    lines += ['', '每格按seed0/1/2排列；“不可行”保留，不以其LUT混入可行均值。全部逐种子、可行标记和层数见old-budget.csv。'
        '预算效应应同时观察早期优势、对照随后追上和共同目标的首次候选数；单看最佳解出现轮次不足以判断。', '',
        '![旧数据预算比较](old-budget.svg)', '', '| 设置 | 电路 | 目标LUT | 组别 | 命中/3 | 首次候选数seed0/1/2 |',
        '|---|---|---:|---|---:|---|']
    for label in ('100x10', '50x50'):
        for name in protocol['circuits']:
            for target in TARGETS:
                for group in GROUPS:
                    bank = [r for r in summary['target_hits'] if r['experiment'] == label and r['circuit'] == name
                            and r['target'] == target and r['group'] == group]
                    times = ' / '.join(str(r['first_candidate']) if r['hit'] else r['censoring'] for r in bank)
                    lines.append(f'| {label} | {name} | {TARGETS[target][name]} ({target}) | {group} | '
                                 f'{sum(r["hit"] for r in bank)}/3 | {times} |')
    lines += ['', '目标要求同时满足原层数限制；未命中记为预算内未达到，未完成记为未完成预算。'
        '不对成功样本单独算平均发现时间，不把未命中当作刚好在预算末尾命中。target-hits.csv保存全部108条记录。', '',
        '## 新50×50搜索及27次运行的层数达标', '',
        '| 电路 | 组别 | seed | 状态 | 最佳LUT/层数 | 动作候选达标 | 轮内有可行解 | 第50步达标 |',
        '|---|---|---:|---|---|---|---|---|']
    for row in summary['training']:
        lines.append(f'| {row["circuit"]} | {row["group"]} | {row["seed"]} | {row["status"]} | '
            f'{row["best_luts"]}/{row["best_levels"]} | {count_rate(row["feasible_candidates"], row["candidates"])} | '
            f'{count_rate(row["feasible_episodes"], row["episodes_completed"])} | '
            f'{count_rate(row["feasible_terminals"], row["episodes_completed"])} |')
    lines += ['', '“轮内有可行解”包含初始映射；候选达标仅统计50个动作后的状态。'
        '搜索允许中间超限，超限结果不会作为可行解比较最佳LUT。', '',
        '| 电路 | 组别 | seed | 最佳LUT | 首次最佳轮次 | 轮内步数 | 累计动作候选 | 最佳动作序列 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for row in summary['training']:
        sequence = ' → '.join((row['best'] or {}).get('sequence', []))
        lines.append(f'| {row["circuit"]} | {row["group"]} | {row["seed"]} | {row["best_luts"]} | '
            f'{row["best_episode"]} | {row["best_iteration"]} | {row["best_candidate"]} | {sequence} |')
    lines += ['', '首次最佳使用与环境一致的可行优先排序；同LUT但更低层数可继续成为新最佳。'
        '不同组各自最佳目标不同，因此“更早达到自己的最佳”不能直接当作更优策略。', '',
        '| 电路 | 对照 | 对照减训练LUT seed0/1/2 | 平均减少LUT | 改善% | 胜/平/负 |',
        '|---|---|---|---:|---:|---|']
    for row in summary['search_comparisons']:
        lines.append(f'| {row["circuit"]} | {row["baseline"]} | {row["differences"]} | '
            f'{format_value(row["mean_luts_reduction"])} | {format_value(row["percent"])} | '
            f'{row["wins"]}/{row["ties"]}/{row["losses"]} |')
    lines += ['', '| 电路 | 新实验候选预算 | trained均值（可行/3） | frozen均值（可行/3） | uniform均值（可行/3） |',
        '|---|---:|---|---|---|']
    for name in protocol['circuits']:
        for budget in (500, 1000, 2500):
            banks = [[r for r in summary['new_budget'] if r['circuit'] == name and r['group'] == group
                      and r['budget'] == budget] for group in GROUPS]
            values = []
            for bank in banks:
                complete = len(bank) == 3 and all(r['status'] == 'complete' for r in bank)
                feasible = sum(bool(r['best_feasible']) for r in bank)
                mean = float(np.mean([r['best_luts'] for r in bank])) if complete and feasible == 3 else None
                values.append(f'{format_value(mean)} ({feasible}/3)')
            lines.append(f'| {name} | {budget} | ' + ' | '.join(values) + ' |')
    lines += ['', '500和1,000候选与旧预算相同，但序列长度和更新次数不同；此处用于定位新设置的搜索曲线，'
        '不能据跨设置差值直接归因于学习。完整逐种子预算切片见search-budget.csv。', '',
        '![2500候选搜索曲线](search-curves.svg)', '', '## 独立评估：主结果与预算前缀', '',
        '| 电路 | 训练seed | 初始化/冻结 | 第25轮 | 第50轮 | uniform共享bank |',
        '|---|---:|---|---|---|---|']
    for name in protocol['circuits']:
        for seed in protocol['seeds']:
            bank = [next(r for r in summary['evaluation'] if r['circuit'] == name and r['training_seed'] == seed
                         and r['policy'] == policy and r['steps'] == 50) for policy in POLICIES]
            lines.append(f'| {name} | {seed} | ' + ' | '.join(
                f'{format_value(r["best_luts_mean"])} LUT；可行{r["feasible"]}/30；完成{r["completed"]}/30' for r in bank) + ' |')
    lines += ['', '以上为完整50步序列找到的最佳解。下表固定使用第50轮模型，前缀均预先指定；'
        '第25轮仅作诊断，不能事后选它替代主结果。', '',
        '| 电路 | 步数 | 对照 | 平均减少LUT及探索性95%区间 | 可行率变化pp及区间 | 可行优先胜/平/负 |',
        '|---|---:|---|---|---|---|']
    for row in summary['comparisons']:
        lines.append(f'| {row["circuit"]} | {row["steps"]} | {row["baseline"]} | '
            f'{format_value(row["mean_luts_reduction"])} {interval(row["ci95"])} | '
            f'{format_value(row["feasibility_gain_pp"])} {interval(row["feasibility_ci95"])} | '
            f'{row["wins"]}/{row["ties"]}/{row["losses"]} |')
    lines += ['', '![固定模型的评估预算前缀](evaluation-prefixes.svg)', '',
        '| 电路 | 步数 | 对照 | 训练seed | LUT减少 | 改善% | 训练/对照可行条数 | 胜/平/负 |',
        '|---|---:|---|---:|---:|---:|---|---|']
    for row in summary['comparisons']:
        for p in row['per_seed']:
            lines.append(f'| {row["circuit"]} | {row["steps"]} | {row["baseline"]} | {p["seed"]} | '
                f'{format_value(p["luts_reduction"])} | {format_value(p["percent"])} | '
                f'{p["trained_feasible"]}/{p["baseline_feasible"]} | {p["wins"]}/{p["ties"]}/{p["losses"]} |')
    lines += ['', '差值为对照减训练，正值表示训练更好。配对分层bootstrap先重采样三个训练种子，再共同重采样配对评估种子列，'
        '保留共同随机数和uniform共享bank的相关性；10,000次，分析种子20260927。'
        '每个前缀不是额外独立实验，90条逻辑评估行不是90次独立训练；所有区间仅作探索性解释，'
        '不据多前缀搜索宣称显著收益，不把未观察到优势写成等效证明。'
        '依据[Agarwal等](https://arxiv.org/abs/2108.13264)。', '',
        '| 电路 | 策略 | seed | 50步终点可行/30 | 终点LUT均值 | 终点层数均值 | 奖励和均值 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in summary['evaluation']:
        if row['steps'] == 50:
            lines.append(f'| {row["circuit"]} | {row["policy"]} | {row["training_seed"]} | {row["terminal_feasible"]}/30 | '
                f'{format_value(row["terminal_luts_mean"])} | {format_value(row["terminal_levels_mean"])} | {format_value(row["reward_mean"])} |')
    lines += ['', '终点LUT只有全部终点可行时才汇总；层数和奖励在完整评估中保留全部序列。', '',
        '## 更新、策略与机制诊断', '',
        '| 电路 | seed | Actor变化RMS | Critic变化RMS | 固定探针KL | 熵 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['diagnostic_summary']:
        if row['group'] == 'trained':
            lines.append(f'| {row["circuit"]} | {row["seed"]} | {row["actor_drift_rms"]:.6f} | '
                f'{row["critic_drift_rms"]:.6f} | {row["policy_kl"]:.6f} | {row["entropy"]:.6f} |')
    lines += ['', 'diagnostics.csv逐轮保存梯度、更新量、损失、回报尺度、参数漂移、固定探针概率、KL及熵。'
        '诊断不消耗采样随机数；探针覆盖局部输入，概率变化不是策略质量改善的充分证据。', '',
        '| 电路 | 组别 | 零奖励比例均值 | LUT往返次数 | 其中两步正奖励次数 |',
        '|---|---|---:|---:|---:|']
    for name in protocol['circuits']:
        for group in GROUPS:
            bank = [r for r in summary['behavior'] if r['circuit'] == name and r['group'] == group]
            if bank:
                lines.append(f'| {name} | {group} | {np.mean([r["zero_reward_fraction"] for r in bank]):.2%} | '
                    f'{sum(r["lut_roundtrips"] for r in bank)} | {sum(r["positive_lut_roundtrips"] for r in bank)} |')
    lines += ['', '观察：每轮初始状态归一化为零，常量特征及持续为零特征名单见summary.json；'
        '减少LUT奖励+3而增加LUT奖励−1，LUT数下降后恢复可能获得两步正奖励。'
        '这里是数值往返，不表示完整网表回到同一状态。奖励与最终最佳LUT错位、状态信息损失、长序列优化难度是候选解释，'
        '本轮没有对它们做干预，不能断言因果。', '', '## 证据、限制与复现', '',
        '训练表只使用checkpoint已提交的完整轮次，逐步重算奖励、回报尺度、达标率、终点与最佳解首次位置。'
        '评估逐条核对50步日志、动作、奖励、最佳值、终点、全部预定前缀和模型哈希。'
        '完整运行的导出最佳映射需有CEC等价记录；映射及未映射网表的最终复核见netlist-validation.json。', '',
        '原100×10数据只读，old-budget.csv与target-hits.csv记录其重新分析；源文件SHA256见analysis-provenance.json。'
        '旧实验哈希保留检查见preserved-inputs.json与preservation-validation.json。源码提交、配置、工具二进制和电路哈希、完整命令、任务状态及时间'
        '见provenance.json和evaluation-provenance.json；失败证据保留于原始结果。', '',
        '每分钟检查任务进度，3个workers、每进程1个Torch线程，单任务硬超时30分钟。'
        '耗时含诊断开销及共享资源并发，本报告用候选预算比较搜索质量。'
        '只有三个训练种子，涉及三个训练电路，未验证未见电路泛化，不宣称10步/50步理论最优。', '',
        '压缩日志、CSV/JSON、曲线和交付哈希保存在本目录；原始权重与完整输出位于results/learning-effectiveness-50x50/。'
        '模型哈希见model-manifest.json；仅凭压缩日志不能重放固定模型，需要复制权重或重新训练。'
        'AI用于实验实现、日志核查、分析及中文报告撰写。', '',
        '```bash', '.tools/conda-env/bin/python -B experiments/learning-effectiveness-50x50/check.py',
        '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-50x50/run.py',
        '# 相同源码、配置、工具及组别指纹才能续跑',
        '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-50x50/run.py --resume',
        '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-50x50/evaluate.py',
        '# 评估中断后同指纹续跑',
        '.tools/conda-env/bin/python -B -u experiments/learning-effectiveness-50x50/evaluate.py --resume',
        '.tools/conda-env/bin/python -B experiments/learning-effectiveness-50x50/verify.py',
        '# 仅从保存证据重建分析报告',
        '.tools/conda-env/bin/python -B experiments/learning-effectiveness-50x50/evaluate.py --report-only', '```', '',
        '在仓库根目录执行。首次执行使用独立干净结果目录；已有相同指纹证据使用--resume。'
        'plot.py在带ReportLab/PyMuPDF的独立Python运行时生成曲线，无需Torch；package.py汇总证据及交付哈希。', '',
        '## 综合判断与下一步', '',
        '判断预算是否掩盖优势，应比较相同目标、相同候选预算下的命中率与首次命中，并核对优势是否随预算增长收敛。'
        '若方向随电路或种子变化，不能仅凭个别“更早找到”归因于预算。'
        '判断长序列学习是否有效，应以第50轮固定策略对初始化和uniform的独立50步评估为主；训练期间找到的单个最佳解只作搜索表现。', '',
        '后续确认需要更多独立训练种子和预先固定的新评估种子。若想区分预算和序列长度的影响，'
        '下一轮保持总候选预算及更新设置可比，分别改变一个因素；若尝试奖励或归一化改进，也采用单因素对照。'
        '本轮无论有无提升都完整保留，不能根据结果换主检查点或只挑有利预算。', '']
    (HERE / 'report.md').write_text('\n'.join(lines))
