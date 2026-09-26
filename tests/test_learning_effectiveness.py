"""Real-tool regressions and semantic checks for the fixed learning experiment."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/learning-effectiveness'))
from common import (ORIGINAL, DiagnosticA2C, config, dump, group_config, load, sha, state_hash)
from drills.model import A2C, ActorCritic, Normalizer
from drills.fpga_session import FPGASession
from evaluate import inference_rollout
from analysis import hierarchical_bootstrap, generate, policy_comparison


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.cfg = config()
        self.cfg['protocol']['episodes'] = 2
        self.cfg['protocol']['iterations'] = 3
        self.circuit = self.cfg['protocol']['circuits']['int2float']

    def tearDown(self):
        self.temp.cleanup()

    def agent(self, label, group='trained', resume=False):
        return DiagnosticA2C(group_config(self.cfg, group, 'test-fingerprint'), self.circuit,
                             0, self.directory / label, resume)

    def test_default_and_diagnostics_match_original_exactly(self):
        source = subprocess.check_output(['git', 'show', f'{ORIGINAL}:drills/model.py'], cwd=ROOT)
        namespace = {'__package__': 'drills', '__name__': 'drills.original_reference'}
        exec(compile(source, 'original-model.py', 'exec'), namespace)
        reference = namespace['A2C'](copy.deepcopy(self.cfg), self.circuit, 0, self.directory / 'reference')
        ordinary = A2C(copy.deepcopy(self.cfg), self.circuit, 0, self.directory / 'ordinary')
        diagnostic = self.agent('diagnostic')
        for _ in range(2):
            reference.run_episode()
            ordinary.run_episode()
            diagnostic.run_episode()
            self.assertEqual(state_hash(reference.network.state_dict()), state_hash(ordinary.network.state_dict()))
            self.assertEqual(state_hash(reference.network.state_dict()), state_hash(diagnostic.network.state_dict()))
            self.assertEqual(state_hash(reference.optimizer.state_dict()), state_hash(diagnostic.optimizer.state_dict()))
            self.assertTrue(torch.equal(reference.rng.get_state(), diagnostic.rng.get_state()))
            ep = reference.episodes_completed
            for name in ('ordinary', 'diagnostic'):
                self.assertEqual((self.directory / f'reference/episodes/{ep}/log.csv').read_bytes(),
                                 (self.directory / f'{name}/episodes/{ep}/log.csv').read_bytes())

    def test_frozen_and_uniform_parameters_optimizer_are_unchanged(self):
        for group in ('frozen', 'uniform'):
            agent = self.agent(group, group)
            initial_network = state_hash(agent.network.state_dict())
            initial_optimizer = state_hash(agent.optimizer.state_dict())
            initial_rng = agent.rng.get_state().clone()
            for _ in range(2):
                agent.run_episode()
                self.assertEqual(initial_network, state_hash(agent.network.state_dict()))
                self.assertEqual(initial_optimizer, state_hash(agent.optimizer.state_dict()))
            self.assertFalse(torch.equal(initial_rng, agent.rng.get_state()))

    def test_first_episode_of_trained_and_frozen_matches(self):
        trained, frozen = self.agent('trained'), self.agent('frozen', 'frozen')
        trained.run_episode()
        frozen.run_episode()
        for name in ('episodes/1/log.csv', 'steps.jsonl'):
            self.assertEqual((self.directory / 'trained' / name).read_bytes(),
                             (self.directory / 'frozen' / name).read_bytes())

    def test_frozen_path_preserves_populated_adam_moments(self):
        cfg = group_config(self.cfg, 'frozen', 'test-fingerprint')
        agent = A2C(cfg, self.circuit, 0, self.directory / 'populated-adam')
        # Exercise nonempty step/exp_avg/exp_avg_sq entries as well as parameter groups.
        for parameter in agent.network.parameters():
            parameter.grad = torch.ones_like(parameter)
        agent.optimizer.step()
        agent.optimizer.zero_grad(set_to_none=True)
        self.assertTrue(agent.optimizer.state_dict()['state'])
        before_network = state_hash(agent.network.state_dict())
        before_optimizer = state_hash(agent.optimizer.state_dict())
        with patch.object(agent, '_update', side_effect=AssertionError('Frozen update called')):
            agent.run_episode()
        self.assertEqual(before_network, state_hash(agent.network.state_dict()))
        self.assertEqual(before_optimizer, state_hash(agent.optimizer.state_dict()))

    def test_real_tool_resume_matches_continuous_for_all_groups(self):
        for group in ('trained', 'frozen', 'uniform'):
            continuous = self.agent(group + '-full', group)
            continuous.run_episode()
            continuous.run_episode()
            interrupted = self.agent(group + '-resumed', group)
            interrupted.run_episode()
            # An interrupted next episode's diagnostic records must not survive resume.
            with (interrupted.folder / 'diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps({'episode': 2, 'partial': True}) + '\n')
            resumed = self.agent(group + '-resumed', group, True)
            resumed.run_episode()
            for field in ('network', 'optimizer', 'rng_state', 'rewards', 'best'):
                self.assertEqual(state_hash(load(continuous.checkpoint)[field]), state_hash(load(resumed.checkpoint)[field]))
            self.assertEqual((continuous.folder / 'steps.jsonl').read_bytes(), (resumed.folder / 'steps.jsonl').read_bytes())
            self.assertEqual((continuous.folder / 'diagnostics.jsonl').read_bytes(), (resumed.folder / 'diagnostics.jsonl').read_bytes())

    def test_resume_rejects_group_fingerprint_and_learning_switch(self):
        self.agent('protected', 'frozen')
        for field, value in [('experiment_group', 'uniform'), ('experiment_fingerprint', 'different')]:
            cfg = group_config(self.cfg, 'frozen', 'test-fingerprint')
            cfg['runtime'][field] = value
            with self.assertRaises(ValueError):
                A2C(cfg, self.circuit, 0, self.directory / 'protected', True)
        with self.assertRaises(ValueError):
            A2C(group_config(self.cfg, 'trained', 'test-fingerprint'), self.circuit, 0,
                self.directory / 'protected', True)

    def test_interruption_between_checkpoint_and_snapshot_is_repaired(self):
        self.cfg['protocol']['episodes'] = 100
        original_write = Path.write_bytes
        def fail_snapshot(path, content):
            if path.parent.name == 'snapshots':
                raise OSError('Injected interruption after checkpoint commit')
            return original_write(path, content)
        with patch.object(Path, 'write_bytes', fail_snapshot):
            with self.assertRaises(OSError):
                self.agent('snapshot-fault')
        agent = self.agent('snapshot-fault', resume=True)
        self.assertEqual(agent.checkpoint.read_bytes(), (agent.folder / 'snapshots/0.pt').read_bytes())
        agent.episodes_completed = 50
        agent.rewards = [0] * 50
        with patch.object(Path, 'write_bytes', fail_snapshot):
            with self.assertRaises(OSError):
                agent.save_model()
        resumed = self.agent('snapshot-fault', resume=True)
        self.assertEqual(resumed.episodes_completed, 50)
        self.assertEqual(resumed.checkpoint.read_bytes(), (resumed.folder / 'snapshots/50.pt').read_bytes())

    def test_uniform_diagnostics_describe_actual_uniform_policy(self):
        agent = self.agent('uniform-entropy', 'uniform')
        agent.run_episode()
        diagnostic = json.loads((agent.folder / 'diagnostics.jsonl').read_text())
        self.assertAlmostEqual(diagnostic['entropy'], np.log(7), places=6)
        np.testing.assert_allclose(diagnostic['probe_probabilities'], 1 / 7, atol=1e-7)
        self.assertEqual(diagnostic['policy_kl'], 0.)

    def test_controlled_positive_advantage_increases_action_probability(self):
        torch.manual_seed(0)
        agent = A2C.__new__(A2C)
        agent.method = self.cfg['method']
        agent.network = ActorCritic(9, 7, agent.method['network'])
        agent.optimizer = torch.optim.Adam(agent.network.parameters(), lr=0.001)
        states = np.ones((10, 9), dtype=np.float32)
        actions = [0] * 5 + [1] * 5
        rewards = [1.] * 5 + [0.] * 5
        before = agent.network(torch.tensor(states))[0].softmax(-1)[0, 0].item()
        agent._update(states, actions, rewards)
        after = agent.network(torch.tensor(states))[0].softmax(-1)[0, 0].item()
        self.assertGreater(after, before)

    def test_normalization_and_reward_diagnostic_properties(self):
        normalizer = Normalizer(9, self.cfg['method']['normalization'])
        np.testing.assert_array_equal(normalizer.normalize(np.ones(9, dtype=np.float32)), np.zeros(9))
        game = FPGASession.__new__(FPGASession)
        game.method, game.circuit = self.cfg['method'], self.circuit
        game.luts, game.levels = 100, 3
        improve = game._get_reward(99, 3)
        game.luts = 99
        self.assertEqual(improve + game._get_reward(100, 3), 2)

    def test_independent_evaluation_does_not_mutate_weights_optimizer_or_rng(self):
        agent = self.agent('training-for-evaluation')
        agent.run_episode()
        before = [state_hash(agent.network.state_dict()), state_hash(agent.optimizer.state_dict()),
                  state_hash(agent.rng.get_state()), state_hash(torch.get_rng_state()), agent.checkpoint.read_bytes()]
        first = inference_rollout(self.cfg, self.circuit, agent.network, 10000, self.directory / 'eval-a')
        second = inference_rollout(self.cfg, self.circuit, agent.network, 10000, self.directory / 'eval-b')
        after = [state_hash(agent.network.state_dict()), state_hash(agent.optimizer.state_dict()),
                 state_hash(agent.rng.get_state()), state_hash(torch.get_rng_state()), agent.checkpoint.read_bytes()]
        self.assertEqual(before, after)
        self.assertEqual(first, second)

    def test_bootstrap_preserves_training_seed_clusters(self):
        matrix = np.array([[0.] * 30, [0.] * 30, [100.] * 30])
        first = hierarchical_bootstrap(matrix)
        self.assertEqual(first, hierarchical_bootstrap(matrix))
        self.assertEqual(first[0], 0)
        self.assertGreaterEqual(first[1], 66.6)
        self.assertEqual(hierarchical_bootstrap(np.ones((3, 30))), [1., 1.])
        shared_bank = np.tile(np.arange(30, dtype=float), (3, 1))
        interval = hierarchical_bootstrap(shared_bank)
        # Three duplicate rows must retain 30-seed uncertainty, not shrink to 90 outcomes.
        self.assertGreater(interval[1] - interval[0], 5.0)
        with self.assertRaises(ValueError):
            hierarchical_bootstrap([[float('nan')]])

    def test_missing_runs_are_retained_in_partial_report(self):
        cfg = config()
        cfg['runtime']['output_dir'] = str(self.directory / 'empty-results')
        destination = self.directory / 'partial-report'
        destination.mkdir()
        with patch('analysis.HERE', destination):
            generate(cfg)
        summary = json.loads((destination / 'summary.json').read_text())
        self.assertEqual(len(summary['training']), 27)
        self.assertEqual(len(summary['evaluation']), 36)
        self.assertEqual(summary['training_complete'], 0)
        self.assertTrue(all(r['mean_luts_reduction'] is None for r in summary['comparisons']))

    def test_failed_partial_training_and_evaluation_are_not_aggregated(self):
        cfg = config()
        root = self.directory / 'failed-results'
        cfg['runtime']['output_dir'] = str(root)
        folder = root / 'training/trained/int2float/seed-0'
        agent = DiagnosticA2C(group_config(cfg, 'trained', 'test'), self.circuit, 0, folder)
        agent.run_episode()
        dump(folder / 'status.json', dict(status='failed', error='Injected failure', episodes=1))
        destination = root / 'evaluation/initial/int2float/seed-0/rollout-10000'
        row = inference_rollout(cfg, self.circuit, agent.reference, 10000, destination)
        row.update(policy='initial', circuit='int2float', training_seed=0, status='complete',
                   checkpoint_sha256=sha(folder / 'snapshots/0.pt'))
        dump(destination / 'rollout.json', row)
        dump(destination.parent / 'status.json', dict(status='failed', completed=1, error='Injected failure'))
        report = self.directory / 'failed-report'
        report.mkdir()
        with patch('analysis.HERE', report):
            generate(cfg)
        summary = json.loads((report / 'summary.json').read_text())
        training = next(r for r in summary['training'] if r['group'] == 'trained' and r['circuit'] == 'int2float' and r['seed'] == 0)
        evaluation = next(r for r in summary['evaluation'] if r['policy'] == 'initial' and r['circuit'] == 'int2float' and r['training_seed'] == 0)
        self.assertEqual((training['status'], training['episodes_completed']), ('failed', 1))
        self.assertEqual((evaluation['status'], evaluation['completed']), ('failed', 1))
        self.assertIsNone(evaluation['best_luts_mean'])
        self.assertTrue(all(r['mean_luts_reduction'] is None and r['wins'] is None for r in summary['comparisons']))

    def test_infeasible_rollouts_are_counted_without_filtered_lut_means(self):
        rows = []
        for seed in (0, 1, 2):
            for policy in ('initial', 'trained-100'):
                for index in range(30):
                    feasible = policy == 'trained-100' or index < 29
                    rows.append(dict(policy=policy, circuit='case', training_seed=seed, evaluation_seed=10000 + index,
                        best_luts=11 if policy == 'trained-100' else 10, best_levels=3 if feasible else 4,
                        best_feasible=feasible, task_status='complete'))
        comparison = policy_comparison(rows, 'case', 'initial', [0, 1, 2])
        self.assertTrue(comparison['complete'])
        self.assertFalse(comparison['complete_feasible'])
        self.assertIsNone(comparison['mean_luts_reduction'])
        self.assertEqual((comparison['baseline_feasible'], comparison['trained_feasible']), (87, 90))
        self.assertAlmostEqual(comparison['feasibility_gain_pp'], 100 / 30)
        self.assertEqual((comparison['wins'], comparison['ties'], comparison['losses']), (3, 0, 87))


if __name__ == '__main__':
    unittest.main()
