"""Actual return targets, learning switch, and isolated experiment continuation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from drills.experiment import load_config
from drills.fpga_session import FPGASession
from drills.model import A2C

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('i2c_return_run', ROOT / 'experiments/i2c-return-handling/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def params():
    return yaml.safe_load((ROOT / 'params.yml').read_text())


class ReturnTests(unittest.TestCase):
    def agent(self, mode):
        agent = A2C.__new__(A2C)
        agent.method = params()['method']
        agent.method['normalization']['returns'] = mode
        return agent

    def test_none_preserves_discounted_targets(self):
        expected = np.array([1 + .99 * 2 + .99**2 * 3, 2 + .99 * 3, 3], dtype=np.float32)
        np.testing.assert_array_equal(self.agent('none')._get_returns([1, 2, 3]), expected)

    def test_standardize_keeps_original_formula(self):
        expected = np.array([1 + .99 * 2 + .99**2 * 3, 2 + .99 * 3, 3], dtype=np.float32)
        expected = (expected - expected.mean()) / max(float(expected.std()), 1e-8)
        np.testing.assert_array_equal(self.agent('standardize')._get_returns([1, 2, 3]), expected)
        np.testing.assert_array_equal(self.agent('standardize')._get_returns([0, 0, 0]), np.zeros(3))

    def test_feasible_reward_scale_reaches_raw_targets(self):
        targets = []
        for scale in [1, 10]:
            cfg = params()
            cfg['method']['reward']['feasible_scale'] = scale
            with tempfile.TemporaryDirectory() as directory:
                game = FPGASession(cfg, cfg['protocol']['circuits']['i2c'], directory)
                game.initial_luts = game.luts = 365
                game.levels = 4
                rewards = []
                for luts in [361, 359, 359, 360]:
                    rewards.append(game._get_reward(luts, 4))
                    game.luts = luts
                targets.append(self.agent('none')._get_returns(rewards))
        np.testing.assert_allclose(targets[1], 10 * targets[0], rtol=1e-6)


class ConfigTests(unittest.TestCase):
    def load(self, cfg):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'params.yml'
            path.write_text(yaml.safe_dump(cfg))
            with patch('drills.experiment.shutil.which', side_effect=lambda value: value):
                return load_config(path)

    def test_learning_flag_and_legacy_default(self):
        self.load(params())
        for value in [True, False]:
            cfg = params()
            cfg['method']['learning_enabled'] = value
            self.load(cfg)
        for value in [0, 1, 'false', None, []]:
            cfg = params()
            cfg['method']['learning_enabled'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'learning_enabled'):
                self.load(cfg)

    def test_return_mode_validation(self):
        for value in ['none', 'standardize']:
            cfg = params()
            cfg['method']['normalization']['returns'] = value
            self.load(cfg)
        for value in ['raw', None, True, []]:
            cfg = params()
            cfg['method']['normalization']['returns'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'normalization.returns'):
                self.load(cfg)

    def test_return_epsilon_validation(self):
        for value in [0, -1, float('nan'), float('inf'), True, None, '1e-8']:
            cfg = params()
            cfg['method']['normalization']['returns_epsilon'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'returns_epsilon'):
                self.load(cfg)


class LearningTests(unittest.TestCase):
    def episode(self, agent):
        with patch.object(agent.game, 'reset', return_value=np.zeros(9, dtype=np.float32)), \
                patch.object(agent.game, 'step', side_effect=[(np.ones(9, dtype=np.float32), .2, False),
                                                              (np.ones(9, dtype=np.float32) * 2, .4, True)]):
            return agent.run_episode()

    def test_frozen_network_optimizer_and_advancing_rng(self):
        cfg = params()
        cfg['method']['learning_enabled'] = False
        cfg['method']['normalization']['returns'] = 'none'
        with tempfile.TemporaryDirectory() as directory:
            agent = A2C(cfg, cfg['protocol']['circuits']['i2c'], 0, directory)
            network = runner.state_hash(agent.network.state_dict())
            optimizer = runner.state_hash(agent.optimizer.state_dict())
            rng = agent.rng.get_state().clone()
            with patch.object(agent, '_update', side_effect=AssertionError('Frozen policy updated')):
                self.episode(agent)
            self.assertEqual(network, runner.state_hash(agent.network.state_dict()))
            self.assertEqual(optimizer, runner.state_hash(agent.optimizer.state_dict()))
            self.assertFalse(torch.equal(rng, agent.rng.get_state()))

    def test_missing_learning_flag_still_updates(self):
        cfg = params()
        with tempfile.TemporaryDirectory() as directory:
            agent = A2C(cfg, cfg['protocol']['circuits']['i2c'], 0, directory)
            before = runner.state_hash(agent.network.state_dict())
            with patch.object(agent, '_update', wraps=agent._update) as update:
                self.episode(agent)
            self.assertEqual(update.call_count, 1)
            self.assertNotEqual(before, runner.state_hash(agent.network.state_dict()))

    def test_checkpoint_rejects_other_group_or_learning_mode(self):
        cfg = params()
        cfg['runtime']['experiment_fingerprint'] = 'A'
        with tempfile.TemporaryDirectory() as directory:
            A2C(cfg, cfg['protocol']['circuits']['i2c'], 0, directory)
            other = copy.deepcopy(cfg)
            other['runtime']['experiment_fingerprint'] = 'B'
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                A2C(other, other['protocol']['circuits']['i2c'], 0, directory, resume=True)
            other = copy.deepcopy(cfg)
            other['method']['learning_enabled'] = False
            with self.assertRaisesRegex(ValueError, 'learning_enabled'):
                A2C(other, other['protocol']['circuits']['i2c'], 0, directory, resume=True)


class RunnerTests(unittest.TestCase):
    def test_group_definitions_and_configuration_fingerprint(self):
        configs = [runner.group_config(group) for group in ['A', 'B', 'C']]
        self.assertEqual(len({c['runtime']['experiment_fingerprint'] for c in configs}), 3)
        self.assertEqual([c['method']['learning_enabled'] for c in configs], [True, True, False])
        self.assertEqual([c['method']['reward']['feasible_scale'] for c in configs], [1, 10, 1])
        for c in configs:
            self.assertEqual(c['method']['normalization']['returns'], 'none')
            self.assertEqual(list(c['protocol']['circuits']), ['i2c'])
        changed = copy.deepcopy(configs[0])
        changed['method']['normalization']['returns'] = 'standardize'
        self.assertNotEqual(runner.fingerprint(changed, 'A'), configs[0]['runtime']['experiment_fingerprint'])

    def test_resume_preflight_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.dump(root / 'experiment.json', dict(fingerprint='A'))
            runner.check_resume(root, 'A', True)
            with self.assertRaises(ValueError):
                runner.check_resume(root, 'B', True)
            with self.assertRaises(FileExistsError):
                runner.check_resume(root, 'A', False)

    @unittest.skipUnless(all((ROOT / '.tools/conda-env/bin' / name).is_file()
                            for name in ['yosys-abc', 'yosys']), 'ABC/Yosys unavailable')
    def test_actual_tools_resume_parity_for_all_groups(self):
        for group in ['A', 'B', 'C']:
            cfg = runner.group_config(group)
            cfg['protocol'].update(episodes=3, iterations=2)
            circuit = cfg['protocol']['circuits']['i2c']
            with self.subTest(group=group), tempfile.TemporaryDirectory() as directory:
                full = A2C(cfg, circuit, 0, Path(directory) / 'full')
                for _ in range(3):
                    full.run_episode()
                partial = A2C(cfg, circuit, 0, Path(directory) / 'split')
                partial.run_episode()
                resumed = A2C(cfg, circuit, 0, Path(directory) / 'split', resume=True)
                for _ in range(2):
                    resumed.run_episode()
                self.assertEqual(full.rewards, resumed.rewards)
                self.assertEqual(full.game.best, resumed.game.best)
                self.assertEqual(runner.state_hash(full.network.state_dict()), runner.state_hash(resumed.network.state_dict()))
                self.assertEqual(runner.state_hash(full.optimizer.state_dict()), runner.state_hash(resumed.optimizer.state_dict()))
                self.assertTrue(torch.equal(full.rng.get_state(), resumed.rng.get_state()))
                for episode in range(1, 4):
                    p = Path('episodes') / str(episode) / 'log.csv'
                    self.assertEqual((full.game.directory / p).read_text(), (resumed.game.directory / p).read_text())


if __name__ == '__main__':
    unittest.main()
