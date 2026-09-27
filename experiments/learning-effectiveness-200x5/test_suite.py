"""5-step real-tool checks plus policy and resume invariants."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import (ROOT, ORIGINAL, SNAPSHOTS, DiagnosticA2C, config, group_config,
                    load, state_hash)
from drills.model import A2C
from evaluate import inference_rollout, physical_seeds, training_validation
from analysis import policy_comparison, hierarchical_bootstrap, generate
from verify import checkpoint_export, verify
from package import validate_coverage
from report import comparison_verdict, circuit_verdict


class ShortExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.cfg = config()
        self.cfg['protocol']['episodes'] = 2
        self.circuit = self.cfg['protocol']['circuits']['int2float']

    def tearDown(self):
        self.temp.cleanup()

    def agent(self, name, group='trained', resume=False):
        return DiagnosticA2C(group_config(self.cfg,group,'test'),self.circuit,0,self.directory/name,resume)

    def test_protocol_only_changes_episode_count_and_length(self):
        cfg = config()
        self.assertEqual((cfg['protocol']['episodes'],cfg['protocol']['iterations']),(200,5))
        self.assertEqual(SNAPSHOTS,(0,100,200))
        self.assertEqual(physical_seeds(cfg,'uniform'),[0])
        self.assertEqual(physical_seeds(cfg,'initial'),[0,1,2])

    def test_diagnostics_match_original_training_at_5_steps(self):
        source = subprocess.check_output(['git','show',f'{ORIGINAL}:drills/model.py'],cwd=ROOT)
        namespace = {'__package__':'drills','__name__':'drills.original_reference'}
        exec(compile(source,'original-model.py','exec'),namespace)
        reference = namespace['A2C'](copy.deepcopy(self.cfg),self.circuit,0,self.directory/'reference')
        diagnostic = self.agent('diagnostic')
        for _ in range(2):
            reference.run_episode()
            diagnostic.run_episode()
            for field in ('network','optimizer','rng_state','rewards','best'):
                self.assertEqual(state_hash(load(reference.checkpoint)[field]),state_hash(load(diagnostic.checkpoint)[field]))
            ep = reference.episodes_completed
            self.assertEqual((self.directory/f'reference/episodes/{ep}/log.csv').read_bytes(),
                             (self.directory/f'diagnostic/episodes/{ep}/log.csv').read_bytes())

    def test_5_step_first_episode_and_old_prefix_agree(self):
        trained,frozen = self.agent('trained'),self.agent('frozen','frozen')
        trained.run_episode(); frozen.run_episode()
        self.assertEqual((trained.folder/'steps.jsonl').read_bytes(),(frozen.folder/'steps.jsonl').read_bytes())
        cfg = copy.deepcopy(self.cfg); cfg['protocol']['iterations']=10
        longer = A2C(cfg,self.circuit,0,self.directory/'longer'); longer.run_episode()
        self.assertEqual(trained.game.log_file.read_text().splitlines(),
                         longer.game.log_file.read_text().splitlines()[:7])

    def test_5_step_continuous_and_resume_match_all_groups(self):
        for group in ('trained','frozen','uniform'):
            full = self.agent(group+'-full',group)
            full.run_episode(); full.run_episode()
            interrupted = self.agent(group+'-resumed',group)
            interrupted.run_episode()
            with (interrupted.folder/'steps.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(episode=2,partial=True))+'\n')
            resumed = self.agent(group+'-resumed',group,True); resumed.run_episode()
            for field in ('network','optimizer','rng_state','rewards','best'):
                self.assertEqual(state_hash(load(full.checkpoint)[field]),state_hash(load(resumed.checkpoint)[field]))
            for name in ('steps.jsonl','diagnostics.jsonl'):
                self.assertEqual((full.folder/name).read_bytes(),(resumed.folder/name).read_bytes())
            if group != 'trained':
                zero = load(full.folder/'snapshots/0.pt'); end = load(full.checkpoint)
                self.assertEqual(state_hash(zero['network']),state_hash(end['network']))
                self.assertEqual(state_hash(zero['optimizer']),state_hash(end['optimizer']))

    def test_frozen_preserves_populated_adam_at_5_steps(self):
        agent = A2C(group_config(self.cfg,'frozen','test'),self.circuit,0,self.directory/'moments')
        for p in agent.network.parameters():
            p.grad=torch.ones_like(p)
        agent.optimizer.step(); agent.optimizer.zero_grad(set_to_none=True)
        before = (state_hash(agent.network.state_dict()),state_hash(agent.optimizer.state_dict()))
        with patch.object(agent,'_update',side_effect=AssertionError('Frozen update called')):
            agent.run_episode()
        self.assertEqual(before,(state_hash(agent.network.state_dict()),state_hash(agent.optimizer.state_dict())))

    def test_evaluation_is_rng_neutral_and_reproducible(self):
        agent = self.agent('train'); agent.run_episode()
        def hashes():
            return [state_hash(agent.network.state_dict()),state_hash(agent.optimizer.state_dict()),
                    state_hash(agent.rng.get_state()),state_hash(torch.get_rng_state()),agent.checkpoint.read_bytes()]
        before = hashes()
        first = inference_rollout(self.cfg,self.circuit,agent.network,20000,self.directory/'eval-first')
        self.assertEqual(before,hashes())
        second = inference_rollout(self.cfg,self.circuit,agent.network,20000,self.directory/'eval-second')
        self.assertEqual(first,second)
        self.assertEqual(len(first['actions']),5)
        self.assertEqual([p['steps'] for p in first['prefixes']],[5])
        self.assertEqual(before,hashes())

    def test_cross_group_fingerprint_and_learning_switch_are_rejected(self):
        self.agent('protected','frozen')
        for key,value in [('experiment_group','uniform'),('experiment_fingerprint','different')]:
            cfg = group_config(self.cfg,'frozen','test'); cfg['runtime'][key]=value
            with self.assertRaises(ValueError):
                A2C(cfg,self.circuit,0,self.directory/'protected',True)
        with self.assertRaises(ValueError):
            self.agent('protected','trained',True)

    def test_all_milestone_snapshots_and_interruption_recovery(self):
        self.cfg['protocol']['episodes']=200
        agent=self.agent('snapshots')
        for ep in SNAPSHOTS:
            agent.episodes_completed=ep; agent.rewards=[0]*ep; agent.save_model()
            self.assertEqual(agent.checkpoint.read_bytes(),(agent.folder/f'snapshots/{ep}.pt').read_bytes())
        target=agent.folder/'snapshots/200.pt'; target.unlink()
        resumed=self.agent('snapshots','trained',True)
        self.assertEqual(resumed.episodes_completed,200)
        self.assertEqual(target.read_bytes(),resumed.checkpoint.read_bytes())

    def test_controlled_long_trajectory_changes_action_probabilities(self):
        agent=self.agent('controlled')
        states=np.ones((5,9),dtype=np.float32)
        before=agent.network(torch.tensor(states))[0].softmax(-1)[0,0].item()
        agent._update(states,[0]*2+[1]*3,[1.]*2+[0.]*3)
        after=agent.network(torch.tensor(states))[0].softmax(-1)[0,0].item()
        self.assertGreater(after,before)

    def evaluation_rows(self):
        return [dict(policy=policy,circuit='case',training_seed=seed,evaluation_seed=20000+i,steps=5,
                     status='complete',task_status='complete',best_luts=10,best_levels=3,best_feasible=True)
                for seed in (0,1,2) for policy in ('initial','trained-200') for i in range(30)]

    def test_infeasible_and_failed_banks_are_not_filtered_or_aggregated(self):
        rows=self.evaluation_rows()
        for r in rows:
            if r['policy']=='initial' and r['evaluation_seed']==20029:
                r.update(best_feasible=False,best_levels=4)
        comparison=policy_comparison(rows,'case','initial',[0,1,2],5)
        self.assertTrue(comparison['complete'])
        self.assertFalse(comparison['complete_feasible'])
        self.assertIsNone(comparison['mean_luts_reduction'])
        self.assertEqual((comparison['baseline_feasible'],comparison['trained_feasible']),(87,90))
        rows=self.evaluation_rows()
        for r in rows:
            if r['policy']=='initial' and r['training_seed']==0:
                r['task_status']='failed'
        comparison=policy_comparison(rows,'case','initial',[0,1,2],5)
        self.assertFalse(comparison['complete'])
        self.assertIsNone(comparison['mean_luts_reduction'])
        self.assertIsNone(comparison['wins'])

    def test_shared_bank_bootstrap_and_missing_evidence(self):
        shared=np.tile(np.arange(30,dtype=float),(3,1))
        interval=hierarchical_bootstrap(shared)
        self.assertEqual(interval,hierarchical_bootstrap(shared))
        self.assertGreater(interval[1]-interval[0],5.)
        cfg=config(); cfg['runtime']['output_dir']=str(self.directory/'missing')
        destination=self.directory/'report'; destination.mkdir()
        with patch('analysis.HERE',destination):
            summary=generate(cfg)
        self.assertEqual(summary['training_complete'],0)
        self.assertEqual(summary['physical_banks_complete'],0)
        self.assertTrue(all(r['mean_luts_reduction'] is None for r in summary['comparisons']))

    def test_partial_checkpoint_fairness_exports_and_timeout_reporting(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['protocol']['circuits']={'int2float':self.circuit}
        cfg['protocol']['seeds']=[0]
        root=self.directory/'partial'; cfg['runtime']['output_dir']=str(root)
        for group in ('trained','frozen','uniform'):
            folder=root/'training'/group/'int2float/seed-0'
            agent=DiagnosticA2C(group_config(cfg,group,'test'),self.circuit,0,folder)
            agent.run_episode()
        validation=training_validation(cfg)
        self.assertEqual((validation['runs'],validation['complete_runs']),(3,0))
        self.assertEqual(len(validation['errors']),3)
        self.assertTrue(validation['first_episodes_equal'])
        self.assertTrue(validation['initial_states_equal'] and validation['frozen_unchanged'])
        self.assertTrue(all(r['episodes_completed']==1 and 200 in r['missing_snapshots'] for r in validation['checks']))
        folder=root/'training/trained/int2float/seed-0'
        expected=checkpoint_export(folder)
        self.assertEqual(verify((cfg,self.circuit,folder,expected,'best-mapped.v','best.v'))['status'],'complete')
        exported=folder/'best.v'; original=exported.read_text(); exported.write_text(original+'\n')
        with self.assertRaises(ValueError):
            checkpoint_export(folder)
        exported.write_text(original)
        processes=root/'logs/training/processes.json'; processes.parent.mkdir(parents=True)
        processes.write_text(json.dumps([dict(label='trained-int2float-0',exit_code=-15,
            timed_out=True,elapsed_seconds=1800)]))
        destination=self.directory/'partial-report'; destination.mkdir()
        with patch('analysis.HERE',destination):
            summary=generate(cfg)
        self.assertEqual(summary['training'][0]['status'],'timed_out')
        self.assertEqual(summary['training_complete'],0)
        self.assertEqual(json.loads((destination/'log-validation.json').read_text())['committed_training_steps'],15)
        self.assertIn('30分钟硬超时',(destination/'report.md').read_text())
        self.assertIn('0/0表示缺失，不表示0%可行',(destination/'report.md').read_text())
        self.assertIn('未完成配对',(destination/'report.md').read_text())

    def test_packaging_partial_evidence_requires_explicit_failures_and_all_cec(self):
        training=dict(status='incomplete',processes=[dict(label=str(i),exit_code=-15 if i<9 else 0) for i in range(27)])
        evaluation=dict(status='incomplete',processes=[dict(label=str(i),exit_code=0) for i in range(27)],skipped=[{}]*3)
        summary=dict(training_complete=18,physical_banks_complete=27,physical_rollouts=810,logical_rollouts=990,
                     training=[dict(candidates=1000)]*18+[dict(candidates=800)]*9)
        validation=dict(evidence_replayed=True,committed_training_steps=25200,physical_evaluation_rollouts=810)
        records=[dict(status='complete',netlist_sha256={'mapped':'hash'})]*3+[
            dict(status='complete',netlist_sha256={'mapped':'hash','raw':'hash'})]*837
        netlists=dict(status='complete',all_present_exports_checked=True,exports_checked=840,
                      expected_present_exports=840,netlists_verified=1677,records=records)
        coverage=validate_coverage(training,evaluation,summary,validation,netlists)
        self.assertFalse(coverage['protocol_complete'])
        self.assertEqual(coverage['evaluation_physical_rollouts'],810)
        with self.assertRaises(ValueError):
            validate_coverage(training,evaluation,summary,dict(validation,evidence_replayed=False),netlists)
        with self.assertRaises(ValueError):
            validate_coverage(training,evaluation,summary,validation,dict(netlists,netlists_verified=1676))
        with self.assertRaises(ValueError):
            validate_coverage(dict(training,status='running'),evaluation,summary,validation,netlists)

    def test_uniform_sampling_probabilities_are_exact_and_frozen(self):
        agent = self.agent('uniform','uniform')
        initial = load(agent.checkpoint)
        agent.run_episode()
        steps = [json.loads(line) for line in (agent.folder/'steps.jsonl').read_text().splitlines()]
        for row in steps:
            np.testing.assert_allclose(row['probabilities'],np.full(7,1/7),rtol=0,atol=1e-8)
        final = load(agent.checkpoint)
        for field in ('network','optimizer'):
            self.assertEqual(state_hash(initial[field]),state_hash(final[field]))

    def test_conclusions_require_both_baselines_intervals_and_seed_consistency(self):
        rows = self.evaluation_rows()
        for r in rows:
            if r['policy']=='trained-200':
                r['best_luts']=9
        comparison = policy_comparison(rows,'case','initial',[0,1,2],5)
        self.assertEqual(comparison_verdict(comparison),'supported_improvement')
        self.assertEqual(circuit_verdict([comparison,comparison]),'本设置下观察到一致收益')
        uncertain = dict(comparison,ci95=[-0.1,2])
        self.assertNotEqual(circuit_verdict([comparison,uncertain]),'本设置下观察到一致收益')
        inconsistent = copy.deepcopy(comparison); inconsistent['per_seed'][0]['luts_reduction']=-1
        self.assertNotEqual(comparison_verdict(inconsistent),'supported_improvement')
        self.assertEqual(comparison_verdict(dict(comparison,complete=False)),'incomplete')

    def test_duplicate_evaluation_seeds_are_rejected(self):
        rows = self.evaluation_rows()
        rows[1]['evaluation_seed']=rows[0]['evaluation_seed']
        with self.assertRaises(ValueError):
            policy_comparison(rows,'case','initial',[0,1,2],5)


if __name__=='__main__':
    unittest.main(verbosity=2)
