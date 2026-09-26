"""Original algorithm equivalence and fixed-budget learning/frozen comparisons."""
import copy
import csv
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drills.fpga_session import FPGASession
from drills.model import A2C

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('learning_horizon_run', ROOT / 'experiments/i2c-learning-horizon/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def historical_module(filename):
    source = subprocess.check_output(['git', 'show', f'{runner.ORIGINAL_COMMIT}:drills/{filename}.py'], cwd=ROOT, text=True)
    module = types.ModuleType('original_' + filename)
    module.__package__ = 'drills'
    exec(compile(source, f'{runner.ORIGINAL_COMMIT}:drills/{filename}.py', 'exec'), module.__dict__)
    return module


class OriginalAlgorithmTests(unittest.TestCase):
    def test_table_rewards_match_original_all_branches(self):
        original = historical_module('fpga_session')
        cfg = runner.group_config('frozen-10')
        with tempfile.TemporaryDirectory() as directory:
            old = original.FPGASession(cfg, cfg['protocol']['circuits']['i2c'], Path(directory) / 'old')
            new = FPGASession(cfg, cfg['protocol']['circuits']['i2c'], Path(directory) / 'new')
            for previous in [3, 4, 5, 6]:
                old.luts = new.luts = 365
                old.levels = new.levels = previous
                for current in [3, 4, 5, 6]:
                    for luts in [360, 364, 365, 366, 370]:
                        with self.subTest(previous=previous, current=current, luts=luts):
                            self.assertEqual(old._get_reward(luts, current), new._get_reward(luts, current))

    def test_standardize_and_update_match_original_exactly(self):
        original = historical_module('model')
        cfg = runner.group_config('trained-30')
        states = np.random.default_rng(7).normal(size=(30, 9)).astype(np.float32)
        actions = [i % 7 for i in range(30)]
        rewards = [3, 0, -1, 3, 3, 0] * 5
        with tempfile.TemporaryDirectory() as directory:
            old = original.A2C(cfg, cfg['protocol']['circuits']['i2c'], 1, Path(directory) / 'old')
            new = A2C(cfg, cfg['protocol']['circuits']['i2c'], 1, Path(directory) / 'new')
            captured = []
            tensor = torch.tensor

            def capture(value, *args, **kwargs):
                array = np.asarray(value)
                if array.shape == (30,) and array.dtype.kind == 'f':
                    captured.append(array.copy())
                return tensor(value, *args, **kwargs)

            with patch.object(original.torch, 'tensor', side_effect=capture):
                old._update(states, actions, rewards)
            self.assertEqual(len(captured), 1)
            np.testing.assert_array_equal(captured[0], new._get_returns(rewards))
            new._update(states, actions, rewards)
            self.assertEqual(runner.state_hash(old.network.state_dict()), runner.state_hash(new.network.state_dict()))
            self.assertEqual(runner.state_hash(old.optimizer.state_dict()), runner.state_hash(new.optimizer.state_dict()))


class ProtocolTests(unittest.TestCase):
    def test_fixed_groups_only_allowed_changes(self):
        configs = {g: runner.group_config(g) for g in runner.GROUPS}
        self.assertEqual(len({c['runtime']['experiment_fingerprint'] for c in configs.values()}), 3)
        for group, cfg in configs.items():
            self.assertEqual(cfg['protocol']['episodes'], 100)
            self.assertEqual(cfg['protocol']['seeds'], [0, 1, 2])
            self.assertEqual(list(cfg['protocol']['circuits']), ['i2c'])
            self.assertEqual(cfg['method']['normalization']['returns'], 'standardize')
            self.assertEqual(cfg['method']['reward']['feasible_mode'], 'table')
            self.assertNotIn('feasible_scale', cfg['method']['reward'])
        trained = copy.deepcopy(configs['trained-30'])
        frozen = copy.deepcopy(configs['frozen-30'])
        trained['method'].pop('learning_enabled')
        frozen['method'].pop('learning_enabled')
        trained.pop('runtime')
        frozen.pop('runtime')
        self.assertEqual(trained, frozen)
        self.assertEqual([c['protocol']['iterations'] for c in configs.values()], [10, 30, 30])

    def test_step_boundary_and_log_count(self):
        for group in ['frozen-10', 'frozen-30']:
            cfg = runner.group_config(group)
            count = cfg['protocol']['iterations']
            with self.subTest(group=group), tempfile.TemporaryDirectory() as directory:
                game = FPGASession(cfg, cfg['protocol']['circuits']['i2c'], directory)

                def fake_run():
                    game.luts, game.levels = 365, 4
                    with game.log_file.open('a', newline='') as stream:
                        csv.writer(stream).writerow([game.iteration, game.sequence[-1], 365, 4, 0])
                    return np.zeros(9, dtype=np.float32), 0

                with patch.object(game, '_run', side_effect=fake_run):
                    game.reset()
                    for step in range(1, count + 1):
                        _, _, done = game.step(0)
                        self.assertEqual(done, step == count)
                    with self.assertRaises(RuntimeError):
                        game.step(0)
                with game.log_file.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(r['iteration']) for r in rows], list(range(count + 1)))

    def test_frozen_state_and_rng_at_both_horizons(self):
        for group in ['frozen-10', 'frozen-30']:
            cfg = runner.group_config(group)
            count = cfg['protocol']['iterations']
            with self.subTest(group=group), tempfile.TemporaryDirectory() as directory:
                agent = A2C(cfg, cfg['protocol']['circuits']['i2c'], 0, directory)
                network = runner.state_hash(agent.network.state_dict())
                optimizer = runner.state_hash(agent.optimizer.state_dict())
                rng = agent.rng.get_state().clone()
                steps = [(np.ones(9, dtype=np.float32) * i, 3, i == count) for i in range(1, count + 1)]
                with patch.object(agent.game, 'reset', return_value=np.zeros(9, dtype=np.float32)), \
                        patch.object(agent.game, 'step', side_effect=steps), \
                        patch.object(agent, '_update', side_effect=AssertionError('Frozen policy updated')):
                    agent.run_episode()
                self.assertEqual(network, runner.state_hash(agent.network.state_dict()))
                self.assertEqual(optimizer, runner.state_hash(agent.optimizer.state_dict()))
                self.assertFalse(torch.equal(rng, agent.rng.get_state()))

    def test_group_and_horizon_resume_rejection(self):
        configs = {g: runner.group_config(g) for g in runner.GROUPS}
        for group, cfg in configs.items():
            with self.subTest(group=group), tempfile.TemporaryDirectory() as directory:
                A2C(cfg, cfg['protocol']['circuits']['i2c'], 0, directory)
                for other, changed in configs.items():
                    if other == group:
                        continue
                    with self.assertRaisesRegex(ValueError, 'learning_enabled|fingerprint'):
                        A2C(changed, changed['protocol']['circuits']['i2c'], 0, directory, resume=True)
                runner.dump(Path(directory) / 'experiment.json', dict(fingerprint=cfg['runtime']['experiment_fingerprint']))
                runner.check_resume(Path(directory), cfg['runtime']['experiment_fingerprint'], True)
                with self.assertRaises(ValueError):
                    runner.check_resume(Path(directory), 'other', True)
                with self.assertRaises(FileExistsError):
                    runner.check_resume(Path(directory), cfg['runtime']['experiment_fingerprint'], False)

    @unittest.skipUnless(all((ROOT / '.tools/conda-env/bin' / name).is_file()
                            for name in ['yosys-abc', 'yosys']), 'ABC/Yosys unavailable')
    def test_actual_tools_resume_parity_all_groups(self):
        for group in runner.GROUPS:
            cfg = runner.group_config(group)
            cfg['protocol']['episodes'] = 2
            cfg['runtime']['experiment_fingerprint'] = runner.fingerprint(cfg, group)
            circuit = cfg['protocol']['circuits']['i2c']
            with self.subTest(group=group), tempfile.TemporaryDirectory() as directory:
                full = A2C(cfg, circuit, 0, Path(directory) / 'full')
                full.run_episode()
                full.run_episode()
                partial = A2C(cfg, circuit, 0, Path(directory) / 'split')
                partial.run_episode()
                resumed = A2C(cfg, circuit, 0, Path(directory) / 'split', resume=True)
                resumed.run_episode()
                self.assertEqual(full.rewards, resumed.rewards)
                self.assertEqual(full.game.best, resumed.game.best)
                self.assertEqual(runner.state_hash(full.network.state_dict()), runner.state_hash(resumed.network.state_dict()))
                self.assertEqual(runner.state_hash(full.optimizer.state_dict()), runner.state_hash(resumed.optimizer.state_dict()))
                self.assertTrue(torch.equal(full.rng.get_state(), resumed.rng.get_state()))
                for episode in [1, 2]:
                    p = Path('episodes') / str(episode) / 'log.csv'
                    self.assertEqual((full.game.directory / p).read_text(), (resumed.game.directory / p).read_text())


if __name__ == '__main__':
    unittest.main()
