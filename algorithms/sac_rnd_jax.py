# Inspired by:
# 1. paper for SAC-RND: https://arxiv.org/abs/2301.13616
# 2. implementation: https://github.com/tinkoff-ai/sac-rnd

# WARN: The ability to save checkpoints is not implemented (for simplicity).
# For more check out flax docs: https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html
import wandb
import uuid
import pyrallis

import jax
import chex
import distrax
import optax
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import gym
import d4rl

import math
from functools import partial
from dataclasses import dataclass, asdict
from flax.core import FrozenDict
from typing import Dict, Tuple, Any, Callable, Sequence
from tqdm.auto import trange
from copy import deepcopy

from flax.training.train_state import TrainState


@dataclass
class Config:
    # wandb params
    project: str = "CORL"
    group: str = "SAC-RND"
    name: str = "SAC-RND"
    # model params
    actor_learning_rate: float = 0.001
    critic_learning_rate: float = 0.001
    alpha_learning_rate: float = 0.001
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_beta: float = 1.0
    critic_beta: float = 1.0
    num_critics: int = 2
    critic_layernorm: bool = True
    # rnd params
    rnd_learning_rate: float = 3e-4
    rnd_hidden_dim: int = 256
    rnd_embedding_dim: int = 32
    rnd_num_epochs: int = 1
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 1024
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 50
    # general params
    train_seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


# source: https://github.com/rail-berkeley/d4rl/blob/d842aa194b416e564e54b0730d9f934e3e32f854/d4rl/__init__.py#L63
# modified to also return next_action (needed for logging mse to dataset actions)
def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i + 1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array]

    @staticmethod
    def create_from_d4rl(dataset_name: str, normalize_reward: bool = False) -> "ReplayBuffer":
        d4rl_data = qlearning_dataset(gym.make(dataset_name))
        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(d4rl_data["next_observations"], dtype=jnp.float32),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32)
        }
        if normalize_reward:
            buffer["rewards"] = ReplayBuffer.normalize_reward(dataset_name, buffer["rewards"])

        return ReplayBuffer(data=buffer)

    @property
    def size(self):
        # WARN: do not use __len__ here! It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError("Reward normalization is implemented only for AntMaze yet!")


@chex.dataclass(frozen=True)
class RunningMeanStd:
    state: Dict[str, jax.Array]

    @staticmethod
    def create(eps: float = 1e-4) -> "RunningMeanStd":
        init_state = {
            "mean": jnp.array([0.0]),
            "var": jnp.array([0.0]),
            "count": jnp.array([eps])
        }
        return RunningMeanStd(state=init_state)

    def update(self, batch: jax.Array) -> "RunningMeanStd":
        batch_mean, batch_var, batch_count = batch.mean(), batch.var(), batch.shape[0]
        if batch_count == 1:
            batch_var = jnp.zeros_like(batch_mean)

        new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
            self.state["mean"], self.state["var"], self.state["count"], batch_mean, batch_var, batch_count
        )
        return self.replace(state={"mean": new_mean, "var": new_var, "count": new_count})

    @staticmethod
    def _update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    @property
    def std(self):
        return jnp.sqrt(self.state["var"])

    @property
    def mean(self):
        return self.state["mean"]


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, float]:
        # cumulative_value / total_steps
        return {k: float(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (arr - mean) / (std + eps)


def pytorch_init(fan_in: float):
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


def identity(x):
    return x


class TorchBilinearDense(nn.Module):
    """
    Implementation of the Bilinear layer as in PyTorch:
    https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html#torch.nn.Bilinear
    """
    out_dim: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, z):
        kernel = self.param(
            'kernel', self.kernel_init, (self.out_dim, x.shape[-1], z.shape[-1]), jnp.float32
        )
        bias = self.param('bias', self.bias_init, (self.out_dim, 1), jnp.float32)
        # with same init and inputs this expression gives all True in torch.isclose for torch.nn.Bilinear
        out = ((x.T * (kernel @ z.T)).sum(1) + bias).T
        return out


class BilinearFirstMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        bilinear = TorchBilinearDense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        combined_emb = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)),
            nn.relu,
            nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        ])
        out = combined_emb(
            nn.relu(bilinear(feature, context))
        )
        return out


class FilmLastMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, feature, context):
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        film = nn.Dense(2 * self.hidden_dim, kernel_init=pytorch_init(c_d), bias_init=pytorch_init(c_d))
        linear1 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d))
        linear2 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear3 = nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))
        linear4 = nn.Dense(self.out_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d))

        gamma, beta = jnp.split(film(context), 2, axis=-1)
        out = nn.relu(linear1(feature))
        out = nn.relu(linear2(out))
        out = nn.relu(gamma * linear3(out) + beta)
        out = linear4(out)
        return out


class RND(nn.Module):
    hidden_dim: int
    embedding_dim: int
    state_mean: jax.Array
    state_std: jax.Array
    action_mean: jax.Array
    action_std: jax.Array

    @nn.compact
    def __call__(self, state, action):
        predictor = BilinearFirstMLP(self.hidden_dim, self.embedding_dim)
        target = FilmLastMLP(self.hidden_dim, self.embedding_dim)

        state = normalize(state, self.state_mean, self.state_std)
        action = normalize(action, self.action_mean, self.action_std)

        pred, target = predictor(action, state), target(action, state)

        return pred, jax.lax.stop_gradient(target)


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


# WARN: only for [-1, 1] action bounds, scaling/unscaling is left as an exercise for the reader :D
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state):
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        net = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
        ])
        log_sigma_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))
        mu_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        trunk = net(state)
        mu, log_sigma = mu_net(trunk), log_sigma_net(trunk)
        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = jnp.clip(log_sigma, -5, 2)

        dist = TanhNormal(mu, jnp.exp(log_sigma))
        return dist


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = False

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        network = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d + a_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ])
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = False

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics
        )
        q_values = ensemble(self.hidden_dim, self.layernorm)(state, action)
        return q_values


class Alpha(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", lambda key: jnp.array([jnp.log(self.init_value)]))
        return jnp.exp(log_alpha)


# SAC-RND losses & update functions
class RNDTrainState(TrainState):
    rms: RunningMeanStd


class CriticTrainState(TrainState):
    target_params: FrozenDict


def rnd_bonus(
        rnd: RNDTrainState,
        state: jax.Array,
        action: jax.Array
) -> jax.Array:
    pred, target = rnd.apply_fn(rnd.params, state, action)
    # [batch_size, embedding_dim]
    bonus = jnp.sum((pred - target) ** 2, axis=1) / rnd.rms.std
    return bonus


def update_rnd(
        key: jax.random.PRNGKey,
        rnd: RNDTrainState,
        batch: Dict[str, jax.Array],
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, RNDTrainState, Metrics]:
    def rnd_loss_fn(params):
        pred, target = rnd.apply_fn(params, batch["states"], batch["actions"])
        raw_loss = ((pred - target) ** 2).sum(axis=1)

        new_rms = rnd.rms.update(raw_loss)
        loss = raw_loss.mean(axis=0)
        return loss, new_rms

    (loss, new_rms), grads = jax.value_and_grad(rnd_loss_fn, has_aux=True)(rnd.params)
    new_rnd = rnd.apply_gradients(grads=grads).replace(rms=new_rms)

    # log rnd bonus for random actions
    key, actions_key = jax.random.split(key)
    random_actions = jax.random.uniform(actions_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
    new_metrics = metrics.update({
        "rnd_loss": loss,
        "rnd_rms": new_rnd.rms.std,
        "rnd_data": loss / rnd.rms.std,
        "rnd_random": rnd_bonus(rnd, batch["states"], random_actions).mean()
    })
    return key, new_rnd, new_metrics


def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        rnd: RNDTrainState,
        critic: TrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        beta: float,
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, actions_key, random_action_key = jax.random.split(key, 3)

    def actor_loss_fn(params):
        actions_dist = actor.apply_fn(params, batch["states"])
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=actions_key)

        rnd_penalty = rnd_bonus(rnd, batch["states"], actions)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) + beta * rnd_penalty - q_values).mean()

        # logging stuff
        actor_entropy = -actions_logp.sum(-1).mean()
        random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
        new_metrics = metrics.update({
            "batch_entropy": actor_entropy,
            "actor_loss": loss,
            "rnd_policy": rnd_penalty.mean(),
            "rnd_random": rnd_bonus(rnd, batch["states"], random_actions).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean()
        })
        return loss, (actor_entropy, new_metrics)

    grads, (actor_entropy, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return key, new_actor, actor_entropy, new_metrics


def update_alpha(
        alpha: TrainState,
        entropy: jax.Array,
        target_entropy: float,
        metrics: Metrics
) -> Tuple[TrainState, Metrics]:
    def alpha_loss_fn(params):
        alpha_value = alpha.apply_fn(params)
        loss = (alpha_value * (entropy - target_entropy)).mean()

        new_metrics = metrics.update({
            "alpha": alpha_value,
            "alpha_loss": loss
        })
        return loss, new_metrics

    grads, new_metrics = jax.grad(alpha_loss_fn, has_aux=True)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)

    return new_alpha, new_metrics


def update_critic(
        key: jax.random.PRNGKey,
        actor: TrainState,
        rnd: RNDTrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        gamma: float,
        beta: float,
        tau: float,
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)

    next_actions_dist = actor.apply_fn(actor.params, batch["next_states"])
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=actions_key)
    rnd_penalty = rnd_bonus(rnd, batch["next_states"], next_actions)

    next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
    next_q = next_q - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1) - beta * rnd_penalty

    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)
    )
    new_metrics = metrics.update({
        "critic_loss": loss,
        "q_min": q_min,
    })
    return key, new_critic, new_metrics


def update_sac(
        key: jax.random.PRNGKey,
        rnd: RNDTrainState,
        actor: TrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: Dict[str, Any],
        target_entropy: float,
        gamma: float,
        actor_beta: float,
        critic_beta: float,
        tau: float,
        metrics: Metrics,
):
    key, new_actor, actor_entropy, new_metrics = update_actor(key, actor, rnd, critic, alpha, batch, actor_beta, metrics)
    new_alpha, new_metrics = update_alpha(alpha, actor_entropy, target_entropy, new_metrics)
    key, new_critic, new_metrics = update_critic(
        key, new_actor, rnd, critic, alpha, batch, gamma, critic_beta, tau, new_metrics
    )
    return key, new_actor, new_critic, new_alpha, new_metrics


# Evaluation
def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        dist = actor.apply_fn(actor.params, obs)
        action = dist.mean()
        return action
    return _action_fn


def evaluate(env: gym.Env, action_fn: Callable, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)

    returns = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = np.asarray(jax.device_get(action_fn(obs)))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


# Training
@pyrallis.wrap()
def train(config: Config):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
    )
    buffer = ReplayBuffer.create_from_d4rl(config.dataset_name, config.normalize_reward)
    state_mean, state_std = buffer.get_moments("states")
    action_mean, action_std = buffer.get_moments("actions")

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, rnd_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)

    eval_env = gym.make(config.dataset_name)
    eval_env.seed(config.eval_seed)

    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    target_entropy = -init_action.shape[-1]

    rnd_module = RND(
        hidden_dim=config.rnd_hidden_dim,
        embedding_dim=config.rnd_embedding_dim,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    rnd = RNDTrainState.create(
        apply_fn=rnd_module.apply,
        params=rnd_module.init(rnd_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.rnd_learning_rate),
        rms=RunningMeanStd.create()
    )
    actor_module = Actor(action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim)
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )
    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=config.alpha_learning_rate)
    )
    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim, num_critics=config.num_critics, layernorm=config.critic_layernorm
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    update_sac_partial = partial(
        update_sac, target_entropy=target_entropy, gamma=config.gamma,
        actor_beta=config.actor_beta, critic_beta=config.critic_beta, tau=config.tau
    )

    def rnd_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_rnd, new_metrics = update_rnd(key, carry["rnd"], batch, carry["metrics"])
        carry.update(
            key=key, rnd=new_rnd, metrics=new_metrics
        )
        return carry

    def sac_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_actor, new_critic, new_alpha, new_metrics = update_sac_partial(
            key=key,
            rnd=carry["rnd"],
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
            metrics=carry["metrics"]
        )
        carry.update(
            key=key, actor=new_actor, critic=new_critic, alpha=new_alpha, metrics=new_metrics
        )
        return carry

    # metrics
    rnd_metrics_to_log = [
        "rnd_loss", "rnd_rms", "rnd_data", "rnd_random"
    ]
    bc_metrics_to_log = [
        "critic_loss", "q_min", "actor_loss", "batch_entropy",
        "rnd_policy", "rnd_random", "action_mse", "alpha_loss", "alpha"
    ]
    # shared carry for update loops
    update_carry = {
        "key": key,
        "actor": actor,
        "rnd": rnd,
        "critic": critic,
        "alpha": alpha,
        "buffer": buffer,
    }
    # PRETRAIN RND
    for epoch in trange(config.rnd_num_epochs, desc="RND Epochs"):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(rnd_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=rnd_loop_update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log({"epoch": epoch, **{f"RND/{k}": v for k, v in mean_metrics.items()}})

    # TRAIN OFFLINE SAC
    for epoch in trange(config.num_epochs, desc="SAC Epochs"):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=sac_loop_update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log({"epoch": epoch, **{f"SAC/{k}": v for k, v in mean_metrics.items()}})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            actor_action_fn = action_fn(actor=update_carry["actor"])

            eval_returns = evaluate(eval_env, actor_action_fn, config.eval_episodes, seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            wandb.log({
                "epoch": epoch,
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })

    wandb.finish()


if __name__ == "__main__":
    train()
