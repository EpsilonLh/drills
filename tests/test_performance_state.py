"""State semantics, original-code parity and real-tool checkpoint integration."""
from copy import deepcopy
import csv
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drills.experiment import load_config
from drills.fpga_session import FPGASession, performance_feature_names
from drills.model import A2C, Normalizer


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = 'adf7bef53cdf6c4161586c03e6e2caecffe26bc4'


def original_class(filename, name):
    source = subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:{filename}'], cwd=ROOT, text=True)
    namespace = {'__name__': 'drills._baseline', '__package__': 'drills'}
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[name], namespace


class PerformanceStateTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'params-new-state.yml')
        self.temporary = tempfile.TemporaryDirectory(prefix='drills-state-tests-')
        self.directory = Path(self.temporary.name)
        self.circuit = self.config['protocol']['circuits']['int2float']

    def tearDown(self):
        self.temporary.cleanup()

    def make_agent(self, config=None, name='agent', resume=False):
        return A2C(config or self.config, self.circuit, 0, self.directory / name, resume=resume)

    def test_configuration_validation(self):
        self.assertEqual(performance_feature_names({}), [])
        self.assertEqual(performance_feature_names({'performance_features': []}), [])
        for invalid in ['lut_ratio', ['unknown'], ['lut_ratio', 'lut_ratio'], [None], [[]]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                performance_feature_names({'performance_features': invalid})
        for maximum in [0, -1, 1.5, True]:
            circuit = dict(self.circuit, max_levels=maximum)
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                FPGASession(self.config, circuit, self.directory / 'invalid')

    def test_feature_values_initial_reference_and_margin_sign(self):
        circuit = dict(self.circuit, max_levels=41)
        game = FPGASession(self.config, circuit, self.directory / 'values')
        readings = iter([(100, 41), (90, 40), (110, 50)])

        def fake_abc(*args, **kwargs):
            luts, levels = next(readings)
            (game.episode_dir / 'current.v').write_text('unmapped')
            (game.episode_dir / 'mapped.v').write_text('mapped')
            return f'nd = {luts} lev = {levels}\n'

        structural = np.arange(9, dtype=np.float32)
        with patch('drills.fpga_session.check_output', side_effect=fake_abc), \
                patch('drills.fpga_session.extract_features', return_value=structural):
            first = game.reset()
            second, _, _ = game.step(0)
            third, _, _ = game.step(0)
        np.testing.assert_array_equal(first[:9], structural)
        np.testing.assert_allclose(first[9:], [1, 0])
        np.testing.assert_allclose(second[9:], [.9, 1 / 41])
        np.testing.assert_allclose(third[9:], [1.1, -9 / 41])
        self.assertEqual(third.dtype, np.float32)
        self.assertEqual(game.initial_luts, 100)

    def test_zero_initial_luts_rejected(self):
        game = FPGASession(self.config, self.circuit, self.directory / 'zero')

        def fake_abc(*args, **kwargs):
            (game.episode_dir / 'current.v').write_text('unmapped')
            (game.episode_dir / 'mapped.v').write_text('mapped')
            return 'nd = 0 lev = 0\n'

        with patch('drills.fpga_session.check_output', side_effect=fake_abc), \
                patch('drills.fpga_session.extract_features', return_value=np.zeros(9, dtype=np.float32)), \
                self.assertRaisesRegex(ValueError, 'initial mapped LUT'):
            game.reset()

    def test_original_normalization_and_network_inputs(self):
        baseline_normalizer, _ = original_class('drills/model.py', 'Normalizer')
        params = self.config['method']['normalization']
        current, baseline = Normalizer(9, params), baseline_normalizer(9, params)
        for row in np.random.default_rng(0).normal(size=(10, 9)).astype(np.float32):
            np.testing.assert_array_equal(current.normalize(row), baseline.normalize(row))
        agent = self.make_agent()
        inputs = []
        handle = agent.network.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
        agent.run_episode()
        handle.remove()
        self.assertEqual(agent.network.actor[0].in_features, 11)
        self.assertEqual(agent.network.critic[0].in_features, 11)
        np.testing.assert_array_equal(inputs[0][:9].numpy(), np.zeros(9, dtype=np.float32))
        np.testing.assert_allclose(inputs[0][9:].numpy(), [1, 0])
        self.assertEqual(inputs[-1].shape, (10, 11))
        with (self.directory / 'agent/episodes/1/log.csv').open(newline='') as handle:
            mapped_steps = list(csv.DictReader(handle))
        initial_luts = int(mapped_steps[0]['luts'])
        for network_input, step in zip(inputs[:-1], mapped_steps[:-1]):
            expected = [int(step['luts']) / initial_luts,
                        (self.circuit['max_levels'] - int(step['levels'])) / self.circuit['max_levels']]
            np.testing.assert_allclose(network_input[9:].numpy(), expected, atol=1e-7)

    def test_no_extra_tool_calls_and_structural_state_parity(self):
        from drills import features, fpga_session
        base = deepcopy(self.config)
        base['method'].pop('performance_features')
        sequences = []
        counts = []
        for config, name in [(base, 'base-tools'), (self.config, 'new-tools')]:
            with patch('drills.fpga_session.check_output', wraps=fpga_session.check_output) as mapping, \
                    patch('drills.features.check_output', wraps=features.check_output) as extracting:
                game = FPGASession(config, self.circuit, self.directory / name)
                states = [game.reset()]
                for action in [0, 6]:
                    states.append(game.step(action)[0])
                sequences.append(states)
                counts.append((mapping.call_count, extracting.call_count))
        self.assertEqual(counts, [(3, 6), (3, 6)])
        for before, after in zip(*sequences):
            np.testing.assert_array_equal(before, after[:9])

    def test_disabled_features_match_original_training_exactly(self):
        baseline_agent, namespace = original_class('drills/model.py', 'A2C')
        baseline_session, _ = original_class('drills/fpga_session.py', 'FPGASession')
        namespace['FPGASession'] = baseline_session
        config = deepcopy(self.config)
        config['method'].pop('performance_features')
        original = baseline_agent(config, self.circuit, 0, self.directory / 'original')
        original.run_episode()
        current = self.make_agent(config, 'disabled')
        current.run_episode()
        self.assertEqual(current.rewards, original.rewards)
        self.assertEqual(current.game.best, original.game.best)
        self.assertTrue(torch.equal(current.rng.get_state(), original.rng.get_state()))
        for name, value in original.network.state_dict().items():
            self.assertTrue(torch.equal(value, current.network.state_dict()[name]), name)
        self.assertEqual((self.directory / 'original/episodes/1/log.csv').read_bytes(),
                         (self.directory / 'disabled/episodes/1/log.csv').read_bytes())
        empty = deepcopy(config)
        empty['method']['performance_features'] = []
        explicit_empty = self.make_agent(empty, 'empty')
        explicit_empty.run_episode()
        for name, value in current.network.state_dict().items():
            self.assertTrue(torch.equal(value, explicit_empty.network.state_dict()[name]), name)

    def test_resume_matches_continuous_training(self):
        continuous = self.make_agent(name='continuous')
        continuous.run_episode()
        continuous.run_episode()
        interrupted = self.make_agent(name='resumed')
        interrupted.run_episode()
        resumed = self.make_agent(name='resumed', resume=True)
        resumed.run_episode()
        self.assertEqual(resumed.rewards, continuous.rewards)
        self.assertEqual(resumed.game.best, continuous.game.best)
        self.assertEqual(resumed.episodes_completed, 2)
        for name, value in continuous.network.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.network.state_dict()[name]), name)

    def test_resume_rejects_changed_feature_definitions(self):
        self.make_agent(name='mismatch')
        changed = deepcopy(self.config)
        changed['method']['performance_features'].reverse()
        with self.assertRaisesRegex(ValueError, 'state features'):
            self.make_agent(changed, 'mismatch', resume=True)
        changed = deepcopy(self.config)
        changed['method']['features'].reverse()
        with self.assertRaisesRegex(ValueError, 'state features'):
            self.make_agent(changed, 'mismatch', resume=True)

    def test_legacy_checkpoint_compatibility(self):
        config = deepcopy(self.config)
        config['method'].pop('performance_features')
        baseline_agent, _ = original_class('drills/model.py', 'A2C')
        baseline_agent(config, self.circuit, 0, self.directory / 'legacy')
        restored = self.make_agent(config, 'legacy', resume=True)
        self.assertEqual(restored.network.actor[0].in_features, 9)
        with self.assertRaisesRegex(ValueError, 'Legacy checkpoints'):
            self.make_agent(name='legacy', resume=True)


if __name__ == '__main__':
    unittest.main()
