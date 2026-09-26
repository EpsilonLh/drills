# Copyright (c) 2019, SCALE Lab, Brown University. BSD-3-Clause; see LICENSE.
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .fpga_session import FPGASession, performance_feature_mask


class Normalizer:
    def __init__(self, size, params):
        self.params = params
        self.n = 0
        self.mean = np.zeros(size)
        self.mean_diff = np.zeros(size)

    def normalize(self, state):
        if self.params['state'] == 'none':
            return state
        self.n += 1
        delta = state - self.mean
        self.mean += delta / self.n
        self.mean_diff += delta * (state - self.mean)
        variance = np.clip(self.mean_diff / self.n, self.params['variance_min'], self.params['variance_max'])
        return ((state - self.mean) / np.sqrt(variance)).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, inputs, actions, params):
        super().__init__()
        activation = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh}[params['activation']]

        def network(hidden, outputs):
            sizes = [inputs] + hidden + [outputs]
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    layers.append(activation())
            return nn.Sequential(*layers)

        self.actor = network(params['actor_hidden'], actions)
        self.critic = network(params['critic_hidden'], 1)
        initialize = {'xavier_uniform': nn.init.xavier_uniform_}[params['initialization']]
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                initialize(layer.weight)
                nn.init.constant_(layer.bias, params['bias_initialization'])

    def forward(self, states):
        return self.actor(states), self.critic(states).squeeze(-1)


class A2C:
    def __init__(self, config, circuit, seed, directory, resume=False):
        self.config = config
        self.method, self.protocol = config['method'], config['protocol']
        torch.set_num_threads(config['environment']['torch_threads'])
        torch.manual_seed(seed)
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.game = FPGASession(config, circuit, directory)
        self.state_features = dict(structural=list(self.method['features']),
                                   performance=list(self.game.performance_features))
        self.performance_mask = performance_feature_mask(self.method)
        inputs = len(self.state_features['structural']) + len(self.state_features['performance'])
        self.network = ActorCritic(inputs, len(self.protocol['actions']),
                                   self.method['network']).to('cpu')
        opt = self.method['optimizer']
        self.optimizer = {'Adam': torch.optim.Adam}[opt['name']](
            self.network.parameters(), lr=opt['learning_rate'], betas=opt['betas'],
            eps=opt['epsilon'], weight_decay=opt['weight_decay'])
        self.checkpoint = Path(directory) / 'checkpoint.pt'
        self.episodes_completed, self.rewards = 0, []
        self.training_seconds = 0.0
        if resume:
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
            saved_features = saved.get('state_features')
            if saved_features is None:
                # Original nine-dimensional checkpoints predate state metadata.
                if self.state_features['performance']:
                    raise ValueError('Legacy checkpoints cannot resume with performance features enabled.')
            elif saved_features != self.state_features:
                raise ValueError('Checkpoint state features do not match the requested configuration.')
            saved_mask = saved.get('performance_feature_mask', [1] * len(self.performance_mask))
            if saved_mask != self.performance_mask:
                raise ValueError('Checkpoint performance feature mask does not match the requested configuration.')
            self.network.load_state_dict(saved['network'])
            current_groups = [dict(g) for g in self.optimizer.param_groups]
            self.optimizer.load_state_dict(saved['optimizer'])
            # Resume optimizer history, but apply the newly requested hyperparameters.
            for group, current in zip(self.optimizer.param_groups, current_groups):
                group.update({k: v for k, v in current.items() if k != 'params'})
            self.rng.set_state(saved['rng_state'])
            self.episodes_completed, self.rewards = saved['episodes_completed'], saved['rewards']
            self.training_seconds = saved.get('training_seconds')
            if self.episodes_completed < 0 or len(self.rewards) != self.episodes_completed:
                raise ValueError('Checkpoint episode count is inconsistent with the prescribed budget.')
            self.game.best, self.game.best_netlists = saved['best'], saved['best_netlists']
            self.game.episode = self.episodes_completed
            self.game.export_best()
        else:
            self.save_model()  # Also permits resuming an interruption during the first episode.

    def run_episode(self):
        if self.episodes_completed >= self.protocol['episodes']:
            raise RuntimeError('The prescribed training budget is already complete.')
        started = perf_counter()
        state = self.game.reset()
        structural_size = len(self.state_features['structural'])
        normalizer = Normalizer(structural_size, self.method['normalization'])
        states, actions, rewards = [], [], []
        done = False
        while not done:
            structural = normalizer.normalize(state[:structural_size])
            # Fixed-scale performance features retain their constraint boundary at zero.
            performance = state[structural_size:]
            if self.performance_mask and not all(self.performance_mask):
                performance = performance * np.asarray(self.performance_mask, dtype=np.float32)
            state = (np.concatenate((structural, performance))
                     if self.state_features['performance'] else structural)
            with torch.no_grad():
                logits, _ = self.network(torch.as_tensor(state, device='cpu'))
                action = torch.multinomial(logits.softmax(-1), 1, generator=self.rng).item()
            states.append(state)
            actions.append(action)
            state, reward, done = self.game.step(action)
            rewards.append(reward)
        self._update(states, actions, rewards)
        if self.training_seconds is not None:
            self.training_seconds += perf_counter() - started
        self.episodes_completed += 1
        self.rewards.append(sum(rewards))
        self.save_model()
        return self.rewards[-1]

    def _update(self, states, actions, rewards):
        returns = np.empty(len(rewards), dtype=np.float32)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = rewards[i] + self.method['gamma'] * cumulative
            returns[i] = cumulative
        norm = self.method['normalization']
        if norm['returns'] == 'standardize':
            returns = (returns - returns.mean()) / max(float(returns.std()), norm['returns_epsilon'])
        logits, values = self.network(torch.tensor(np.asarray(states), device='cpu'))
        advantage = torch.tensor(returns, device='cpu') - values
        log_probs = logits.log_softmax(-1).gather(1, torch.tensor(actions, device='cpu')[:, None]).squeeze(1)
        settings = self.method['loss']
        actor = getattr(-log_probs * advantage.detach(), settings['reduction'])()
        critic = getattr(advantage.square(), settings['reduction'])()
        loss = settings['actor_weight'] * actor + settings['critic_weight'] * critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        temporary = self.checkpoint.with_suffix('.tmp')
        torch.save(dict(network=self.network.state_dict(),
                        optimizer=self.optimizer.state_dict(), rng_state=self.rng.get_state(),
                        episodes_completed=self.episodes_completed, rewards=self.rewards,
                        state_features=self.state_features,
                        performance_feature_mask=self.performance_mask,
                        training_seconds=self.training_seconds,
                        best=self.game.best, best_netlists=self.game.best_netlists), temporary)
        temporary.replace(self.checkpoint)
        self.game.export_best()
