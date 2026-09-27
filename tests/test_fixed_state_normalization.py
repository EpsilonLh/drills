"""Semantic, backward-compatible and real-tool normalization regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/fixed-state-normalization'))
from study import (A2C, DiagnosticA2C, Normalizer, ActorCritic, config, group_config, state_hash,
                   load, sha, inference_rollout, read_rows)
from results import reductions, log_replay


class FixedNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.cfg = config()
        self.cfg['protocol']['episodes'] = 2
        self.cfg['protocol']['iterations'] = 3
        self.circuit = self.cfg['protocol']['circuits']['i2c']
        self.features = self.cfg['method']['features']
        self.norm = self.cfg['method']['normalization']

    def tearDown(self):
        self.temp.cleanup()

    def test_fixed_scale_counts_zero_and_fractions(self):
        initial = np.asarray([10,2,100,200,8,0,.6,.1,.3],dtype=np.float32)
        state = np.asarray([10,2,80,220,6,3,.7,.2,.1],dtype=np.float32)
        normalizer = Normalizer(9,self.norm,self.features,initial)
        expected = np.asarray([1,1,.8,1.1,.75,3,.7,.2,.1],dtype=np.float32)
        np.testing.assert_array_equal(normalizer.normalize(state),expected)
        self.assertEqual(normalizer.normalize(state).dtype,np.float32)
        self.assertGreater(normalizer.normalize(state)[3],1)

    def test_same_state_is_independent_of_path_and_reference_is_copied(self):
        initial=np.asarray([10,2,100,200,8,0,.6,.1,.3],dtype=np.float32)
        a=Normalizer(9,self.norm,self.features,initial)
        b=Normalizer(9,self.norm,self.features,initial)
        target=initial*.8
        a.normalize(initial*1.2)
        b.normalize(initial*.3)
        before=a.normalize(target).copy()
        initial[:]=999
        np.testing.assert_array_equal(a.normalize(target),b.normalize(target))
        np.testing.assert_array_equal(a.normalize(target),before)

    def test_feature_order_and_invalid_inputs(self):
        names=['not_fraction','cells','latches']
        norm=Normalizer(3,self.norm,names,np.asarray([.3,100,0]))
        np.testing.assert_array_equal(norm.normalize([.2,50,2]),np.asarray([.2,.5,2],dtype=np.float32))
        for names,state in [(['unknown'],[1]),(['cells','cells'],[1,2]),(['cells'],[float('nan')])]:
            with self.assertRaises(ValueError): Normalizer(len(names),self.norm,names,state)
        with self.assertRaises(ValueError): Normalizer(9,self.norm)
        with self.assertRaises(ValueError): norm.normalize([1,2])
        with self.assertRaises(ValueError): norm.normalize([float('inf'),1,1])

    def test_legacy_welford_is_exactly_unchanged(self):
        params=copy.deepcopy(self.norm);params['state']='episode_welford'
        modern=Normalizer(9,params,self.features,np.ones(9))
        # Independently retain the prior implementation from the pinned commit.
        import subprocess
        source=subprocess.check_output(['git','show','341a8db:drills/model.py'],cwd=ROOT)
        namespace={'__package__':'drills','__name__':'drills.normalizer_reference'}
        exec(compile(source,'reference-model.py','exec'),namespace)
        legacy=namespace['Normalizer'](9,params)
        for values in [np.arange(9,dtype=np.float32),np.arange(9,dtype=np.float32)*2,np.arange(9,dtype=np.float32)*.3]:
            np.testing.assert_array_equal(modern.normalize(values),legacy.normalize(values))

    def agent(self,name,group='fixed-trained',resume=False,production=False):
        cls=A2C if production else DiagnosticA2C
        return cls(group_config(self.cfg,group,'test-fingerprint'),self.circuit,0,self.folder/name,resume)

    def test_production_and_diagnostics_match_for_both_modes(self):
        for group in ('old-trained','fixed-trained'):
            ordinary=self.agent(group+'-base',group,production=True)
            diagnostic=self.agent(group+'-diagnostic',group)
            for episode in (1,2):
                ordinary.run_episode();diagnostic.run_episode()
                for key in ('network','optimizer','rng_state','rewards','best'):
                    self.assertEqual(state_hash(load(ordinary.checkpoint)[key]),state_hash(load(diagnostic.checkpoint)[key]))
                self.assertEqual((ordinary.checkpoint.parent/f'episodes/{episode}/log.csv').read_bytes(),
                                 (diagnostic.folder/f'episodes/{episode}/log.csv').read_bytes())
            best,curve,rewards=log_replay(ordinary.config,diagnostic.folder,2,read_rows(diagnostic.folder/'steps.jsonl'))
            self.assertEqual(best,ordinary.game.best)
            self.assertEqual(rewards,ordinary.rewards)
            self.assertEqual(len(curve),7)

    def test_fixed_training_frozen_pair_and_reference(self):
        trained=self.agent('trained'); frozen=self.agent('frozen','fixed-frozen')
        initial=load(frozen.checkpoint)
        trained.run_episode();frozen.run_episode()
        self.assertEqual((trained.folder/'episodes/1/log.csv').read_bytes(),(frozen.folder/'episodes/1/log.csv').read_bytes())
        self.assertEqual((trained.folder/'steps.jsonl').read_bytes(),(frozen.folder/'steps.jsonl').read_bytes())
        frozen.run_episode()
        final=load(frozen.checkpoint)
        for key in ('network','optimizer'):self.assertEqual(state_hash(initial[key]),state_hash(final[key]))
        rows=read_rows(frozen.folder/'steps.jsonl')
        self.assertEqual(len({tuple(r['reference_state']) for r in rows}),1)

    def test_real_tool_resume_matches_continuous(self):
        for group in ('fixed-trained','fixed-frozen'):
            full=self.agent(group+'-full',group);full.run_episode();full.run_episode()
            partial=self.agent(group+'-resume',group);partial.run_episode()
            with (partial.folder/'steps.jsonl').open('a') as stream:stream.write(json.dumps({'episode':2,'partial':True})+'\n')
            resumed=self.agent(group+'-resume',group,True);resumed.run_episode()
            for key in ('network','optimizer','rng_state','rewards','best','state_normalization'):
                self.assertEqual(state_hash(load(full.checkpoint)[key]),state_hash(load(resumed.checkpoint)[key]))
            for name in ('steps.jsonl','diagnostics.jsonl'):
                self.assertEqual((full.folder/name).read_bytes(),(resumed.folder/name).read_bytes())

    def test_resume_rejects_mode_feature_order_and_legacy_fixed(self):
        self.agent('protected')
        for change in ('mode','order'):
            cfg=group_config(self.cfg,'fixed-trained','test-fingerprint')
            if change=='mode':cfg['method']['normalization']['state']='episode_welford'
            else:cfg['method']['features']=list(reversed(self.features))
            with self.assertRaises(ValueError):A2C(cfg,self.circuit,0,self.folder/'protected',True)
        saved=load(self.folder/'protected/checkpoint.pt');saved.pop('state_normalization')
        torch.save(saved,self.folder/'protected/checkpoint.pt')
        with self.assertRaises(ValueError):self.agent('protected',resume=True)

    def test_inference_uses_training_coordinates_and_is_immutable(self):
        agent=self.agent('model','fixed-frozen');agent.run_episode()
        cfg=group_config(self.cfg,'fixed-frozen')
        digest=sha(agent.checkpoint);before=state_hash(agent.network.state_dict())
        row=inference_rollout(cfg,agent.network,30000,self.folder/'inference')
        self.assertEqual(sha(agent.checkpoint),digest)
        self.assertEqual(before,state_hash(agent.network.state_dict()))
        norm=Normalizer(9,self.norm,self.features,row['reference_state'])
        for step in row['steps']:
            np.testing.assert_array_equal(norm.normalize(step['raw_state']),np.asarray(step['normalized_state'],dtype=np.float32))
        self.assertEqual(row['reference_state'],read_rows(agent.folder/'steps.jsonl')[0]['reference_state'])

    def test_learning_and_representation_are_separate_comparisons(self):
        # Both frozen and trained improve by five LUTs: engineering gain, no extra learning gain.
        values={k:np.asarray(v,dtype=float) for k,v in {
            'old-trained':[100,101,102], 'old-frozen':[102,103,104],
            'fixed-trained':[95,96,97], 'fixed-frozen':[97,98,99]}.items()}
        result=reductions(values)
        self.assertEqual(result['engineering']['mean'],5)
        self.assertEqual(result['learning_change']['mean'],0)


if __name__=='__main__':unittest.main()
