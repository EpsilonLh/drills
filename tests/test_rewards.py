"""Reward contracts and deterministic continuation of a real ABC/Yosys search."""
import copy
from pathlib import Path
import re
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


def config():
    return yaml.safe_load((ROOT / 'params.yml').read_text())


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = config()
        self.session = FPGASession(self.config, self.config['protocol']['circuits']['int2float'],
                                   self.temp.name)
        self.session.initial_luts = 49
        self.session.luts, self.session.levels = 49, 3

    def test_magnitude_symmetry_and_zero(self):
        self.assertAlmostEqual(self.session._get_reward(48, 3), 1 / 49)
        self.assertAlmostEqual(self.session._get_reward(46, 3), 3 / 49)
        self.assertAlmostEqual(self.session._get_reward(52, 3), -3 / 49)
        self.assertEqual(self.session._get_reward(49, 3), 0)
        self.config['method']['reward']['feasible_scale'] = 10
        self.assertAlmostEqual(self.session._get_reward(46, 3), 30 / 49)

    def test_round_trip_has_zero_raw_reward(self):
        first = self.session._get_reward(50, 3)
        self.session.luts = 50
        second = self.session._get_reward(49, 3)
        self.assertEqual(first + second, 0)

    def test_feasible_boundary_includes_equality(self):
        for previous, current in [(2, 3), (3, 3), (3, 2)]:
            with self.subTest(previous=previous, current=current):
                self.session.levels = previous
                self.assertAlmostEqual(self.session._get_reward(48, current), 1 / 49)

    def test_entering_feasible_region_keeps_table(self):
        self.session.levels = 4
        for luts, expected in [(48, 3), (49, 0), (50, -1)]:
            with self.subTest(luts=luts):
                self.assertEqual(self.session._get_reward(luts, 3), expected)

    def test_all_infeasible_table_entries_and_leaving_feasibility(self):
        # Columns: LUT worsening, unchanged, improving. Rows: depth worsening, same, improving.
        expected = {-1: [-3, -2, -1], 0: [-2, 0, 2], 1: [1, 2, 3]}
        for previous, depth in [(4, -1), (5, 0), (6, 1)]:
            self.session.levels = previous
            for luts, reward in zip([50, 49, 48], expected[depth]):
                with self.subTest(previous=previous, luts=luts):
                    self.assertEqual(self.session._get_reward(luts, 5), reward)
        self.session.levels = 3
        for luts, reward in zip([50, 49, 48], expected[-1]):
            self.assertEqual(self.session._get_reward(luts, 4), reward)

    def test_table_mode_and_legacy_missing_mode(self):
        for mode in ['table', None]:
            if mode is None:
                self.config['method']['reward'].pop('feasible_mode')
                self.config['method']['reward'].pop('feasible_scale')
            else:
                self.config['method']['reward']['feasible_mode'] = mode
            for previous in [2, 3, 4, 5, 6]:
                self.session.levels = previous
                for current in [2, 3, 4, 5, 6]:
                    for luts in [48, 49, 50]:
                        area = int(luts < 49) - int(luts > 49)
                        depth = int(current < previous) - int(current > previous)
                        table = self.config['method']['reward']
                        expected = table['feasible'][area] if current <= 3 else table['infeasible'][depth][area]
                        with self.subTest(mode=mode, previous=previous, current=current, luts=luts):
                            self.assertEqual(self.session._get_reward(luts, current), expected)

    def fake_abc(self, metrics):
        metrics = iter(metrics)

        def run(command, **kwargs):
            for filename in re.findall(r'write_verilog "([^"]+)"', command[-1]):
                Path(filename).write_text('module fake(); endmodule\n')
            luts, levels = next(metrics)
            return f'nd = {luts} lev = {levels}\n'

        return run

    def test_initial_luts_fixed_then_reset_for_next_episode(self):
        with patch('drills.fpga_session.check_output', self.fake_abc([(49, 3), (50, 3), (49, 3),
                                                                  (98, 3), (97, 3)])), \
                patch('drills.fpga_session.extract_features', return_value=np.zeros(9)):
            self.session.reset()
            self.assertEqual(self.session.initial_luts, 49)
            _, first, _ = self.session.step(0)
            self.assertEqual(self.session.initial_luts, 49)
            _, second, _ = self.session.step(0)
            self.assertEqual(first + second, 0)
            self.session.reset()
            self.assertEqual(self.session.initial_luts, 98)
            _, reward, _ = self.session.step(0)
            self.assertAlmostEqual(reward, 1 / 98)

    def test_zero_initial_luts_rejected_only_in_delta_mode(self):
        with patch('drills.fpga_session.check_output', self.fake_abc([(0, 0), (0, 0)])), \
                patch('drills.fpga_session.extract_features', return_value=np.zeros(9)):
            with self.assertRaisesRegex(ValueError, 'positive initial mapped LUTs'):
                self.session.reset()
            self.config['method']['reward']['feasible_mode'] = 'table'
            self.session.reset()
            self.assertEqual(self.session.initial_luts, 0)


class ConfigTests(unittest.TestCase):
    def load(self, reward):
        params = config()
        params['method']['reward'] = reward
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'params.yml'
            filename.write_text(yaml.safe_dump(params))
            with patch('drills.experiment.shutil.which', side_effect=lambda value: value):
                return load_config(filename)

    def test_valid_modes_and_legacy_defaults(self):
        reward = config()['method']['reward']
        self.load(reward)
        reward['feasible_mode'] = 'table'
        self.load(reward)
        reward.pop('feasible_mode')
        reward.pop('feasible_scale')
        loaded = self.load(reward)
        self.assertEqual(loaded['method']['reward'], reward)

    def test_invalid_mode(self):
        for mode in ['delta', None, True, 1, []]:
            reward = config()['method']['reward']
            reward['feasible_mode'] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'feasible_mode'):
                self.load(reward)

    def test_invalid_scale(self):
        for scale in [0, -1, float('inf'), float('-inf'), float('nan'), True, None, '1']:
            reward = config()['method']['reward']
            reward['feasible_scale'] = scale
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'feasible_scale'):
                self.load(reward)


class ResumeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(all((ROOT / '.tools/conda-env/bin' / name).is_file()
                            for name in ['yosys-abc', 'yosys']), 'Local ABC/Yosys tools unavailable')
    def test_resumed_search_matches_uninterrupted_search(self):
        params = config()
        params['protocol'].update(episodes=4, iterations=4)
        for key, name in [('abc_binary', 'yosys-abc'), ('yosys_binary', 'yosys')]:
            params['runtime'][key] = str(ROOT / '.tools/conda-env/bin' / name)
        circuit = copy.deepcopy(params['protocol']['circuits']['int2float'])
        circuit['file'] = str(ROOT / circuit['file'])
        with tempfile.TemporaryDirectory() as directory:
            full = A2C(params, circuit, 1, Path(directory) / 'full')
            for _ in range(4):
                full.run_episode()
            split = A2C(params, circuit, 1, Path(directory) / 'split')
            for _ in range(2):
                split.run_episode()
            resumed = A2C(params, circuit, 1, Path(directory) / 'split', resume=True)
            for _ in range(2):
                resumed.run_episode()
            self.assertEqual(full.rewards, resumed.rewards)
            self.assertEqual(full.game.best, resumed.game.best)
            self.assertEqual(resumed.game.initial_luts, 49)
            self.assertTrue(torch.equal(full.rng.get_state(), resumed.rng.get_state()))
            for name, value in full.network.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.network.state_dict()[name]), name)
            for episode in range(1, 5):
                relative = Path('episodes') / str(episode) / 'log.csv'
                self.assertEqual((full.game.directory / relative).read_text(),
                                 (resumed.game.directory / relative).read_text())


if __name__ == '__main__':
    unittest.main()
