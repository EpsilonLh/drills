# Copyright (c) 2019, SCALE Lab, Brown University. BSD-3-Clause; see LICENSE.
import csv
import json
from pathlib import Path
import re
from subprocess import check_output

import numpy as np

from .features import extract_features


def performance_feature_names(method):
    names = method.get('performance_features', [])
    if (not isinstance(names, list)
            or any(name not in ('lut_ratio', 'depth_margin') for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('performance_features must be a list of distinct lut_ratio/depth_margin names.')
    return list(names)


class FPGASession:
    def __init__(self, config, circuit, directory):
        self.config, self.circuit = config, circuit
        self.protocol, self.method = config['protocol'], config['method']
        self.performance_features = performance_feature_names(self.method)
        if self.performance_features and (type(circuit['max_levels']) is not int or circuit['max_levels'] <= 0):
            raise ValueError('Performance features require a positive integer max_levels.')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.episode = 0
        self.best, self.best_netlists = None, {}

    def reset(self):
        self.episode += 1
        self.iteration = 0
        self.sequence = list(self.protocol['initial_sequence'])
        self.luts = self.levels = float('inf')
        self.initial_luts = None
        self.episode_dir = self.directory / 'episodes' / str(self.episode)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.episode_dir / 'log.csv'
        self.log_file.write_text('iteration,optimization,luts,levels,reward\n')
        state, _ = self._run()
        return state

    def step(self, action):
        if not self.episode or self.iteration >= self.protocol['iterations']:
            raise RuntimeError('Reset before starting another episode.')
        self.iteration += 1
        self.sequence.append(self.protocol['actions'][action])
        state, reward = self._run()
        return state, reward, self.iteration == self.protocol['iterations']

    @staticmethod
    def rank(result):
        if result['feasible']:
            return (0, result['luts'], result['levels'])
        return (1, result['levels'], result['luts'])

    def _run(self):
        unmapped, mapped = self.episode_dir / 'current.v', self.episode_dir / 'mapped.v'
        # A failed command must never reuse a previous step's netlist.
        unmapped.unlink(missing_ok=True)
        mapped.unlink(missing_ok=True)
        command = f'read "{self.circuit["file"]}"; ' + '; '.join(self.sequence) + '; '
        command += f'write_verilog "{unmapped}"; if -K {self.protocol["lut_inputs"]}; '
        command += f'write_verilog "{mapped}"; print_stats;'
        output = check_output([self.config['runtime']['abc_binary'], '-c', command], text=True)
        if not unmapped.is_file() or not mapped.is_file():
            raise RuntimeError('ABC did not produce the current netlists.\n' + output)
        luts, levels = map(int, re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)[-1])
        state = extract_features(unmapped, self.config)
        if self.performance_features:
            if self.iteration == 0:
                if luts <= 0:
                    raise ValueError('Performance features require a positive initial mapped LUT count.')
                self.initial_luts = luts
            performance = dict(lut_ratio=luts / self.initial_luts,
                               depth_margin=(self.circuit['max_levels'] - levels) / self.circuit['max_levels'])
            state = np.concatenate((state, np.asarray(
                [performance[name] for name in self.performance_features], dtype=np.float32)))
        reward = self._get_reward(luts, levels) if self.iteration else 0
        self.luts, self.levels = luts, levels
        with self.log_file.open('a', newline='') as log:
            csv.writer(log).writerow([self.iteration, self.sequence[-1], luts, levels, reward])
        result = dict(luts=luts, levels=levels, feasible=levels <= self.circuit['max_levels'],
                      sequence=list(self.sequence), episode=self.episode, iteration=self.iteration)
        eligible = self.iteration > 0 or self.protocol['evaluation']['include_initial']
        if eligible and (self.best is None or self.rank(result) < self.rank(self.best)):
            self.best = result
            self.best_netlists = {'best.v': unmapped.read_text(), 'best-mapped.v': mapped.read_text()}
        return state, reward

    def export_best(self):
        # Called at checkpoint boundaries, and on restore; partial episodes cannot enter the score.
        for name in ['best.v', 'best-mapped.v', 'best.json']:
            if self.best is None:
                (self.directory / name).unlink(missing_ok=True)
            elif name == 'best.json':
                (self.directory / name).write_text(json.dumps(self.best, indent=2) + '\n')
            else:
                (self.directory / name).write_text(self.best_netlists[name])

    def _get_reward(self, luts, levels):
        area = int(luts < self.luts) - int(luts > self.luts)
        if levels <= self.circuit['max_levels']:
            return self.method['reward']['feasible'][area]
        depth = int(levels < self.levels) - int(levels > self.levels)
        return self.method['reward']['infeasible'][depth][area]
