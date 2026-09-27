"""Fixed protocol, instrumented original A2C, and bounded process supervision."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from drills.experiment import load_config
from drills.model import A2C, ActorCritic, Normalizer

ORIGINAL = 'adf7bef53cdf6c4161586c03e6e2caecffe26bc4'
GROUPS = ('trained', 'frozen', 'uniform')
POLICIES = ('initial', 'trained-25', 'trained-50', 'uniform')
EVAL_SEEDS = list(range(10000, 10030))
SNAPSHOTS = (0, 10, 25, 50)
EVAL_PREFIXES = (5, 10, 20, 30, 40, 50)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    temporary.replace(path)


def state_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def sources():
    paths = [ROOT / 'drills.py', ROOT / 'requirements.txt',
             *sorted((ROOT / 'drills').glob('*.py')), HERE / 'common.py', HERE / 'run.py', HERE / 'protocol.yml']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def config():
    cfg = load_config(HERE / 'protocol.yml')
    reference = yaml.safe_load(subprocess.check_output(['git', 'show', f'{ORIGINAL}:params.yml'], cwd=ROOT))
    # Only episode count and trajectory length differ; all algorithm settings are pinned.
    for key in ('protocol', 'method', 'environment'):
        expected = copy.deepcopy(reference[key])
        if key == 'protocol':
            expected.update(episodes=50, iterations=50)
            for circuit in expected['circuits'].values():
                circuit['file'] = str((ROOT / circuit['file']).resolve())
        if cfg[key] != expected:
            raise ValueError(f'Protocol unexpectedly differs from the original: {key}')
    if cfg['runtime']['workers'] != 3:
        raise ValueError('This suite prescribes three workers.')
    return cfg


def identity(cfg):
    tools = {k: sha(cfg['runtime'][k]) for k in ('abc_binary', 'yosys_binary')}
    benchmarks = {k: sha(v['file']) for k, v in cfg['protocol']['circuits'].items()}
    payload = dict(sources=sources(), config=cfg, tools=tools, benchmarks=benchmarks,
                   python=sys.version, torch=torch.__version__, numpy=np.__version__)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def group_config(cfg, group, fingerprint):
    cfg = copy.deepcopy(cfg)
    cfg['method']['learning_enabled'] = group == 'trained'
    cfg['runtime'].update(output_dir=str(Path(cfg['runtime']['output_dir']) / 'training' / group),
                          experiment_group=group, experiment_fingerprint=fingerprint)
    return cfg


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def flattened(module):
    return torch.cat([p.detach().flatten().clone() for p in module.parameters()])


def rms(tensor):
    return float(tensor.double().square().mean().sqrt())


def returns_for(rewards, method):
    result = np.empty(len(rewards), dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + method['gamma'] * cumulative
        result[i] = cumulative
    processed = result.copy()
    if method['normalization']['returns'] == 'standardize':
        processed = (processed - processed.mean()) / max(float(processed.std()), method['normalization']['returns_epsilon'])
    return result, processed


class DiagnosticA2C(A2C):
    """Same sampling/update order as A2C; only deterministic observations added."""
    def __init__(self, cfg, circuit, seed, directory, resume=False):
        self.folder = Path(directory)
        self.group = cfg['runtime']['experiment_group']
        super().__init__(cfg, circuit, seed, directory, resume)
        snapshot = self.folder / 'snapshots' / f'{self.episodes_completed}.pt'
        if resume and self.episodes_completed in SNAPSHOTS and not snapshot.exists():
            # A checkpoint can commit before its milestone copy; recover that exact state.
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            temporary = snapshot.with_suffix('.tmp')
            temporary.write_bytes(self.checkpoint.read_bytes())
            temporary.replace(snapshot)
        initial = load(self.folder / 'snapshots/0.pt')
        self.initial_network = initial['network']
        self.initial_optimizer_hash = state_hash(initial['optimizer'])
        # Deepcopy avoids consuming either global or local sampling RNG.
        self.reference = copy.deepcopy(self.network)
        self.reference.load_state_dict(self.initial_network)
        for name in ('steps.jsonl', 'diagnostics.jsonl'):
            path = self.folder / name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows
                                        if r['episode'] <= self.episodes_completed))

    def save_model(self):
        super().save_model()
        if self.episodes_completed in SNAPSHOTS:
            target = self.folder / 'snapshots' / f'{self.episodes_completed}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(self.checkpoint.read_bytes())
            temporary.replace(target)

    def run_episode(self):
        if self.episodes_completed >= self.protocol['episodes']:
            raise RuntimeError('The prescribed budget is complete.')
        started = time.perf_counter()
        state = self.game.reset()
        normalizer = Normalizer(len(state), self.method['normalization'])
        states, actions, rewards, steps = [], [], [], []
        before = {name: flattened(getattr(self.network, name)) for name in ('actor', 'critic')}
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
                normalized_state=states[-1].tolist(), probabilities=probs.tolist(), value=float(value)))
        raw_returns, processed = returns_for(rewards, self.method)
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
            if self.group == 'uniform':
                initial_logp = torch.full_like(initial_logp, -np.log(initial_logp.shape[-1]))
                current_logp = initial_logp.clone()
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
            diagnostic[name + '_update_rms'] = rms(flattened(module) - before[name])
            diagnostic[name + '_drift_rms'] = rms(flattened(module) - flattened(getattr(self.reference, name)))
        if not self.method.get('learning_enabled', True):
            if state_hash(self.network.state_dict()) != state_hash(self.initial_network) or state_hash(self.optimizer.state_dict()) != self.initial_optimizer_hash:
                raise RuntimeError('Frozen network or Adam changed.')
        for name, rows in [('steps.jsonl', steps), ('diagnostics.jsonl', [diagnostic])]:
            with (self.folder / name).open('a') as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.save_model()
        return self.rewards[-1]


def supervise(tasks, output, resume=False, workers=3, timeout=1800):
    """Run each task once; record failures and continue the rest without retries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending, active, records = list(tasks), [], []
    last_update = 0
    while pending or active:
        while pending and len(active) < workers:
            label, command = pending.pop(0)
            path = output / f'{label}.log'
            stream = path.open('a' if resume else 'w')
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            active.append(dict(label=label, command=command, log=path, stream=stream, process=process,
                               started=now(), start=time.monotonic(), timed_out=False))
        for job in list(active):
            process = job['process']
            if process.poll() is None and time.monotonic() - job['start'] > timeout:
                print(f'Hard timeout: stopping {job["label"]} after {timeout}s.', flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                job['timed_out'] = True
            code = process.poll()
            if code is not None:
                job['stream'].close()
                row = dict(label=job['label'], command=job['command'], started_at=job['started'],
                    finished_at=now(), elapsed_seconds=time.monotonic() - job['start'], exit_code=code,
                    timed_out=job['timed_out'], log=str(job['log'].relative_to(ROOT)), log_sha256=sha(job['log']))
                records.append(row)
                active.remove(job)
                dump(output / 'processes.json', records)
                print(f'Finished {job["label"]}: exit={code}; completed {len(records)}/{len(tasks)}.', flush=True)
        if time.monotonic() - last_update >= 60:
            row = dict(time=now(), completed=len(records), total=len(tasks),
                       active=[dict(label=j['label'], pid=j['process'].pid,
                                    elapsed_seconds=time.monotonic() - j['start'], log_bytes=j['log'].stat().st_size)
                               for j in active])
            with (output / 'monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(f'Progress: {len(records)}/{len(tasks)} finished; active: {[j["label"] for j in active]}', flush=True)
            last_update = time.monotonic()
        if active:
            time.sleep(1)
    return records
