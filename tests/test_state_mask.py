"""Matched initialization, feature masking and old eleven-feature compatibility."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from drills.experiment import load_config
from drills.fpga_session import performance_feature_mask
from drills.model import A2C


ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_COMMIT = 'e1a99bf'


class StateMaskTests(unittest.TestCase):
    def setUp(self):
        self.full = load_config(ROOT / 'params-new-state.yml')
        self.zero = deepcopy(self.full)
        self.zero['method']['performance_feature_mask'] = [0, 0]
        self.circuit = self.full['protocol']['circuits']['int2float']
        self.temporary = tempfile.TemporaryDirectory(prefix='drills-mask-tests-')
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def agent(self, config, name, seed=0, resume=False):
        return A2C(config, self.circuit, seed, self.directory / name, resume=resume)

    def previous_agent(self, name):
        source = subprocess.check_output(['git', 'show', f'{PERFORMANCE_COMMIT}:drills/model.py'],
                                         cwd=ROOT, text=True)
        namespace = {'__name__': 'drills._previous_state', '__package__': 'drills'}
        exec(compile(source, 'previous_model.py', 'exec'), namespace)
        return namespace['A2C'](self.full, self.circuit, 0, self.directory / name)

    def test_mask_validation(self):
        self.assertEqual(performance_feature_mask(self.full['method']), [1, 1])
        self.assertEqual(performance_feature_mask({}), [])
        for mask in [None, [0], [0, 0, 0], [2, 0], [True, 0], [1.0, 0], '00']:
            method = dict(self.full['method'], performance_feature_mask=mask)
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                performance_feature_mask(method)

    def test_initial_weights_and_sampling_rng_match_for_all_ten_seeds(self):
        for seed in range(10):
            full = self.agent(self.full, f'full-{seed}', seed)
            zero = self.agent(self.zero, f'zero-{seed}', seed)
            self.assertEqual(full.network.actor[0].in_features, 11)
            self.assertEqual(zero.network.critic[0].in_features, 11)
            self.assertEqual(sum(v.numel() for v in full.network.parameters()), 938)
            for name, value in full.network.state_dict().items():
                self.assertTrue(torch.equal(value, zero.network.state_dict()[name]), (seed, name))
            self.assertTrue(torch.equal(full.rng.get_state(), zero.rng.get_state()))

    def test_zero_features_reach_both_networks_and_input_weights_do_not_update(self):
        zero = self.agent(self.zero, 'zero')
        before = {name: value.clone() for name, value in zero.network.state_dict().items()}
        observed = []
        handle = zero.network.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
        zero.run_episode()
        handle.remove()
        self.assertEqual(len(observed), 11)
        for state in observed:
            self.assertTrue(torch.equal(state[..., 9:], torch.zeros_like(state[..., 9:])))
        for name in ['actor.0.weight', 'critic.0.weight']:
            self.assertTrue(torch.equal(before[name][:, 9:], zero.network.state_dict()[name][:, 9:]))
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in zero.network.state_dict().items()))

    def test_unmasked_training_matches_previous_eleven_feature_code_exactly(self):
        previous = self.previous_agent('previous')
        previous.run_episode()
        current = self.agent(self.full, 'current')
        current.run_episode()
        self.assertEqual(current.rewards, previous.rewards)
        self.assertEqual(current.game.best, previous.game.best)
        for name, value in previous.network.state_dict().items():
            self.assertTrue(torch.equal(value, current.network.state_dict()[name]), name)
        self.assertTrue(torch.equal(previous.rng.get_state(), current.rng.get_state()))
        self.assertEqual((self.directory / 'previous/episodes/1/log.csv').read_bytes(),
                         (self.directory / 'current/episodes/1/log.csv').read_bytes())

    def test_masked_resume_matches_continuous_training_and_rejects_mask_changes(self):
        continuous = self.agent(self.zero, 'continuous')
        continuous.run_episode()
        continuous.run_episode()
        interrupted = self.agent(self.zero, 'resumed')
        interrupted.run_episode()
        resumed = self.agent(self.zero, 'resumed', resume=True)
        resumed.run_episode()
        self.assertEqual(continuous.rewards, resumed.rewards)
        self.assertEqual(continuous.game.best, resumed.game.best)
        for name, value in continuous.network.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.network.state_dict()[name]), name)
        with self.assertRaisesRegex(ValueError, 'feature mask'):
            self.agent(self.full, 'resumed', resume=True)

    def test_previous_eleven_feature_checkpoint_defaults_to_unmasked_only(self):
        self.previous_agent('legacy')
        self.agent(self.full, 'legacy', resume=True)
        with self.assertRaisesRegex(ValueError, 'feature mask'):
            self.agent(self.zero, 'legacy', resume=True)


if __name__ == '__main__':
    unittest.main()
