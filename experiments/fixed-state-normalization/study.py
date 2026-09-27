"""Locked single-factor normalization study; historical helpers stay immutable."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('normalization_history', ROOT / 'experiments/learning-effectiveness/common.py')
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
from drills.model import A2C, ActorCritic, Normalizer
from drills.experiment import load_config, verify_netlist
from drills.fpga_session import FPGASession

sha, dump, load, state_hash, now = history.sha, history.dump, history.load, history.state_hash, history.now
BASE = '341a8db73a9dbec46bcd10eff31c9bdb0ee6c970'
GROUPS = ('old-trained', 'old-frozen', 'fixed-trained', 'fixed-frozen', 'uniform')
NEW_GROUPS = ('fixed-trained', 'fixed-frozen')
SEEDS = (0, 1, 2)
EVAL_SEEDS = tuple(range(30000, 30030))


def config():
    cfg = load_config(HERE / 'protocol.yml')
    expected = yaml.safe_load(subprocess.check_output(['git', 'show', BASE + ':experiments/learning-effectiveness/protocol.yml'], cwd=ROOT))
    expected['protocol']['circuits'] = {'i2c': expected['protocol']['circuits']['i2c']}
    expected['protocol']['circuits']['i2c']['file'] = str(ROOT / 'benchmarks/i2c.aig')
    expected['method']['normalization']['state'] = 'fixed_initial'
    for key in ('protocol', 'method', 'environment'):
        if cfg[key] != expected[key]:
            raise ValueError('Study differs from locked original settings: ' + key)
    if cfg['runtime']['workers'] != 3 or cfg['experiment'] != {
            'evaluation_seeds': list(EVAL_SEEDS), 'timeout_seconds': 1800,
            'minimum_mean_lut_reduction': 1, 'minimum_seed_wins': 2}:
        raise ValueError('Study execution or screening settings changed.')
    return cfg


def sources():
    paths = [ROOT / 'requirements.txt', *sorted((ROOT / 'drills').glob('*.py')),
             ROOT / 'experiments/learning-effectiveness/common.py',
             HERE / 'protocol.yml', *sorted(HERE.glob('*.py')),
             ROOT / 'tests/test_fixed_state_normalization.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def identity(cfg):
    payload = dict(sources=sources(), config=cfg,
                   tools={k: sha(cfg['runtime'][k]) for k in ('abc_binary', 'yosys_binary')},
                   benchmark=sha(cfg['protocol']['circuits']['i2c']['file']),
                   python=sys.version, torch=torch.__version__, numpy=np.__version__)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def group_config(cfg, group, fingerprint=None):
    cfg = copy.deepcopy(cfg)
    cfg['method']['normalization']['state'] = 'fixed_initial' if group.startswith('fixed-') else 'episode_welford'
    cfg['method']['learning_enabled'] = group.endswith('-trained')
    cfg['runtime'].update(experiment_group=group, experiment_fingerprint=fingerprint)
    return cfg


def assert_history_unchanged():
    pinned = json.loads((HERE / 'historical-inputs.json').read_text())
    bad = [p for p, h in pinned['files'].items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    if bad:
        raise ValueError('Historical files changed: ' + repr(bad[:10]))
    return len(pinned['files'])


def prepare(cfg):
    root = Path(cfg['runtime']['output_dir'])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a fresh result directory; prepare never overwrites a run.')
    count = assert_history_unchanged()
    old_here = ROOT / 'experiments/learning-effectiveness'
    provenance = json.loads((old_here / 'provenance.json').read_text())
    for path, expected in provenance['sources'].items():
        data = subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Historical source does not match its recorded commit: ' + path)
    fingerprint, payload = identity(cfg)
    for key, digest in payload['tools'].items():
        if digest != provenance['tools'][key]:
            raise ValueError('Historical tool differs: ' + key)
    if payload['benchmark'] != provenance['benchmarks']['i2c']:
        raise ValueError('Historical benchmark differs.')
    for key in ('python', 'torch', 'numpy'):
        if payload[key] != provenance[key]:
            raise ValueError('Historical environment differs: ' + key)
    old_root = ROOT / 'results/learning-effectiveness'
    model_hashes = json.loads((old_here / 'model-manifest.json').read_text())['hashes']
    model_hashes = {p:h for p,h in model_hashes.items() if '/i2c/' in p}
    for p,h in model_hashes.items():
        if sha(old_root / p) != h:
            raise ValueError('Historical model changed: ' + p)
    imports = []
    for group, old_group in [('old-trained', 'trained'), ('old-frozen', 'frozen'), ('uniform', 'uniform')]:
        for seed in SEEDS:
            source = old_root / 'training' / old_group / 'i2c' / f'seed-{seed}'
            target = root / 'training' / group / 'i2c' / f'seed-{seed}'
            result = json.loads((source / 'result.json').read_text())
            if result['status'] != 'complete' or result['episodes_completed'] != 100:
                raise ValueError('Incomplete historical result.')
            shutil.copytree(source, target)
            imports.append(dict(group=group, seed=seed, source=str(source),
                                result_sha256=sha(source / 'result.json'), source_fingerprint=provenance['fingerprint']))
    shutil.copytree(old_root / 'baseline/i2c', root / 'baseline/i2c')
    dump(root / 'baseline.json', {'i2c': json.loads((old_root / 'baseline.json').read_text())['i2c']})
    dump(root / 'experiment.json', dict(fingerprint=fingerprint, **payload, started_at=now(),
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        base_commit=BASE, imports=imports, historical_files=count, historical_models=len(model_hashes),
        training_candidates_new=6000, training_candidates_reused=9000, evaluation_candidates=3900))
    print('Prepared six new searches and nine historical controls.', flush=True)


def guard(cfg):
    root = Path(cfg['runtime']['output_dir'])
    manifest = json.loads((root / 'experiment.json').read_text())
    if identity(cfg)[0] != manifest['fingerprint']:
        raise ValueError('Source/configuration/environment fingerprint changed after protocol lock.')
    return root, manifest


class DiagnosticA2C(history.DiagnosticA2C):
    """Original sampling/update order with explicit initial-state normalization."""
    def run_episode(self):
        if self.episodes_completed >= self.protocol['episodes']:
            raise RuntimeError('The prescribed budget is complete.')
        started = time.perf_counter()
        state = self.game.reset()
        initial_state = state.copy()
        normalizer = Normalizer(len(state), self.method['normalization'], self.method['features'], initial_state)
        states, actions, rewards, steps = [], [], [], []
        before = {name: history.flattened(getattr(self.network, name)) for name in ('actor', 'critic')}
        done = False
        while not done:
            raw_state = state.copy()
            state = normalizer.normalize(state)
            with torch.no_grad():
                logits, value = self.network(torch.as_tensor(state, device='cpu'))
                probs = logits.softmax(-1) if self.group != 'uniform' else torch.full_like(logits, 1 / len(logits))
                action = torch.multinomial(probs, 1, generator=self.rng).item()
            states.append(state)
            actions.append(action)
            state, reward, done = self.game.step(action)
            rewards.append(reward)
            steps.append(dict(episode=self.episodes_completed + 1, iteration=self.game.iteration,
                action=action, optimization=self.protocol['actions'][action], luts=self.game.luts,
                levels=self.game.levels, reward=reward, raw_state=raw_state.tolist(),
                normalized_state=states[-1].tolist(), reference_state=initial_state.tolist(),
                probabilities=probs.tolist(), value=float(value)))
        raw_returns, processed = history.returns_for(rewards, self.method)
        with torch.no_grad():
            logits, values = self.network(torch.tensor(np.asarray(states), device='cpu'))
            advantage = torch.tensor(processed, device='cpu') - values
            logp = logits.log_softmax(-1).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
            reduction = self.method['loss']['reduction']
            actor_loss = float(getattr(-logp * advantage, reduction)())
            critic_loss = float(getattr(advantage.square(), reduction)())
        if self.method.get('learning_enabled', True):
            self._update(states, actions, rewards)
        self.training_seconds += time.perf_counter() - started
        self.episodes_completed += 1
        self.rewards.append(sum(rewards))
        probes_file = self.folder / 'probes.pt'
        if not probes_file.exists():
            torch.save(torch.tensor(np.asarray(states)), probes_file)
        probes = load(probes_file)
        with torch.no_grad():
            initial_logp = self.reference(probes)[0].log_softmax(-1)
            current_logp = self.network(probes)[0].log_softmax(-1)
            current_probs = current_logp.exp()
            kl = float((initial_logp.exp() * (initial_logp - current_logp)).sum(-1).mean())
            entropy = float((-current_probs * current_logp).sum(-1).mean())
        diagnostic = dict(episode=self.episodes_completed, actor_loss=actor_loss, critic_loss=critic_loss,
            raw_returns_std=float(raw_returns.std()), processed_returns_std=float(processed.std()),
            zero_rewards=sum(r == 0 for r in rewards), reward_sum=sum(rewards),
            policy_kl=max(0.0, kl), entropy=entropy, probe_probabilities=current_probs.tolist())
        for name in ('actor', 'critic'):
            module = getattr(self.network, name)
            gradient = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
            diagnostic[name + '_gradient_norm'] = float(torch.cat(gradient).norm()) if gradient else 0.0
            diagnostic[name + '_update_rms'] = history.rms(history.flattened(module) - before[name])
            diagnostic[name + '_drift_rms'] = history.rms(history.flattened(module) - history.flattened(getattr(self.reference, name)))
        if not self.method.get('learning_enabled', True):
            if state_hash(self.network.state_dict()) != state_hash(self.initial_network) or state_hash(self.optimizer.state_dict()) != self.initial_optimizer_hash:
                raise RuntimeError('Frozen network or Adam changed.')
        for name, rows in [('steps.jsonl', steps), ('diagnostics.jsonl', [diagnostic])]:
            with (self.folder / name).open('a') as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.save_model()
        return self.rewards[-1]


def train_task(cfg, group, seed, resume=False):
    root, manifest = guard(cfg)
    folder = root / 'training' / group / 'i2c' / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    local = group_config(cfg, group, manifest['fingerprint'])
    agent = DiagnosticA2C(local, local['protocol']['circuits']['i2c'], seed, folder,
                          resume and (folder / 'checkpoint.pt').exists())
    for _ in range(agent.episodes_completed, cfg['protocol']['episodes']):
        reward = agent.run_episode()
        print(f'{group}/seed-{seed}: {agent.episodes_completed}/100, reward={reward}', flush=True)
    verify_netlist(local, local['protocol']['circuits']['i2c'], folder / 'best-mapped.v', agent.game.best)
    dump(folder / 'result.json', dict(group=group, circuit='i2c', seed=seed, status='complete',
        episodes_completed=agent.episodes_completed, best=agent.game.best,
        rewards=agent.rewards, training_seconds=agent.training_seconds))


def inference_rollout(cfg, network, seed, folder, uniform=False):
    game = FPGASession(cfg, cfg['protocol']['circuits']['i2c'], folder)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    state = game.reset()
    initial = state.copy()
    normalizer = Normalizer(len(state), cfg['method']['normalization'], cfg['method']['features'], initial)
    steps, actions, rewards = [], [], []
    done = False
    while not done:
        raw = state.copy()
        state = normalizer.normalize(state)
        with torch.no_grad():
            logits, _ = network(torch.as_tensor(state, device='cpu'))
            probs = logits.softmax(-1) if not uniform else torch.full_like(logits, 1 / len(logits))
            action = torch.multinomial(probs, 1, generator=generator).item()
        steps.append(dict(raw_state=raw.tolist(), normalized_state=state.tolist(), probabilities=probs.tolist()))
        state, reward, done = game.step(action)
        actions.append(action)
        rewards.append(reward)
    game.export_best()
    verify_netlist(cfg, cfg['protocol']['circuits']['i2c'], Path(folder) / 'best-mapped.v', game.best)
    return dict(evaluation_seed=seed, actions=actions, rewards=rewards, steps=steps,
                reference_state=initial.tolist(), best=game.best,
                terminal=dict(luts=game.luts, levels=game.levels, feasible=game.levels <= 4))


def evaluate_task(cfg, group, seed, resume=False):
    root, _ = guard(cfg)
    folder = root / 'evaluation' / group / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    checkpoint = root / 'training' / group / 'i2c' / f'seed-{seed}/checkpoint.pt'
    digest = sha(checkpoint)
    saved = load(checkpoint)
    local = group_config(cfg, group)
    expected = dict(mode=local['method']['normalization']['state'], features=local['method']['features'])
    if group.startswith('fixed-') and saved.get('state_normalization') != expected:
        raise ValueError('Evaluation normalization differs from training.')
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(9, 7, local['method']['network'])
    network.load_state_dict(saved['network'])
    network.eval().requires_grad_(False)
    before = state_hash(network.state_dict())
    rows = []
    for evaluation_seed in EVAL_SEEDS:
        destination = folder / f'rollout-{evaluation_seed}'
        record = destination / 'rollout.json'
        if resume and record.exists():
            row = json.loads(record.read_text())
            if row['checkpoint_sha256'] != digest or row['status'] != 'complete':
                raise ValueError('Different or incomplete evaluation checkpoint.')
        else:
            row = inference_rollout(local, network, evaluation_seed, destination, group == 'uniform')
            row.update(group=group, training_seed=seed, checkpoint_sha256=digest, status='complete')
            dump(record, row)
        rows.append(row)
        print(f'{group}/{seed}: rollout {len(rows)}/30, LUT={row["best"]["luts"]}', flush=True)
    if before != state_hash(network.state_dict()) or sha(checkpoint) != digest:
        raise RuntimeError('Evaluation changed the model.')
    dump(folder / 'result.json', dict(status='complete', group=group, seed=seed,
         checkpoint_sha256=digest, network_unchanged=True, checkpoint_unchanged=True, rollouts=rows))


def execute(cfg, phase, resume=False):
    root, _ = guard(cfg)
    groups = NEW_GROUPS if phase == 'train' else GROUPS
    tasks = []
    for group in groups:
        for seed in ((0,) if phase == 'evaluate' and group == 'uniform' else SEEDS):
            command = [sys.executable, '-B', '-u', str(HERE / 'run.py'), phase, '--task', group, str(seed)]
            if resume:
                command.append('--resume')
            tasks.append((f'{group}-{seed}', command))
    records = history.supervise(tasks, root / 'logs' / phase, resume, workers=3, timeout=1800)
    successful = all(r['exit_code'] == 0 for r in records)
    dump(root / (phase + '-provenance.json'), dict(started_tasks=len(tasks), processes=records,
         finished_at=now(), status='complete' if successful else 'incomplete'))
    if not successful:
        raise RuntimeError('Some prescribed tasks failed; failures retained, no automatic retry.')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
