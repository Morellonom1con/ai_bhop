import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import time

from bhop_gym_env import BhopEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Policy/Value net for vector obs + continuous actions ---
class MlpPolicy(nn.Module):
    def __init__(self, in_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, act_dim)                 # action means
        self.log_std = nn.Parameter(torch.zeros(act_dim)) # learnable log-std
        self.v = nn.Linear(128, 1)                        # state value
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.v(x)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)  # diagonal Gaussian
        return dist, value

# --- Simple FIFO buffer yielding minibatches ---
class ReplayBuffer:
    def __init__(self, keys, buffer_size, mini_batch_size, device):
        self.data_keys = keys
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.data = {k: deque(maxlen=buffer_size) for k in keys}

    def reset(self):
        for k in self.data_keys:
            self.data[k].clear()

    def data_log(self, name, t):
        # split by batch dimension into individual samples
        self.data[name].extend(t.cpu().split(1))

    def __iter__(self):
        n = len(self.data[self.data_keys[0]])
        n = n - n % self.mini_batch_size
        idx = np.random.permutation(n)
        idx_chunks = np.split(idx, n // self.mini_batch_size)
        for ids in idx_chunks:
            batch = {}
            for k in self.data_keys:
                batch[k] = torch.cat([self.data[k][i] for i in ids]).to(self.device)
            batch["batch_size"] = len(ids)
            yield batch

# --- PPO losses ---
def ppo_actor_loss(new_dist, actions, old_logp, adv, clip_eps):
    # Normal.log_prob returns [B, A]; sum over action dims → [B, 1]
    new_logp = new_dist.log_prob(actions).sum(dim=1, keepdim=True)
    ratio = (new_logp - old_logp).exp()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.min(surr1, surr2).mean()  # minus because we minimize loss

def ppo_critic_loss(new_v, old_v, ret, clip_eps):
    v1 = (new_v - ret).pow(2.)
    v_clipped = old_v + torch.clamp(new_v - old_v, -clip_eps, clip_eps)
    v2 = (v_clipped - ret).pow(2.)
    return torch.max(v1, v2).mean()

# --- GAE ---
def compute_gae(next_v, rewards, masks, values, gamma=0.99, lam=0.95):
    gae = 0
    returns, advs = deque(), deque()
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_v * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        next_v = values[t]
        returns.appendleft(gae + values[t])
        advs.appendleft(gae)
    return returns, advs

# --- Hyperparams ---
lr = 3e-4
num_steps = 256          # rollout length
num_mini_batch = 8
ppo_epochs = 3
gamma = 0.99
lam = 0.95
clip_eps = 0.2
max_frames = int(1e6)

# --- Env / model ---
env = BhopEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]  # Box(6,)

model = MlpPolicy(obs_dim, act_dim).to(device)
optim_ = optim.Adam(model.parameters(), lr=lr)

buffer_size = num_steps
mini_batch_size = buffer_size // num_mini_batch
keys = ["states", "actions", "log_probs", "values", "returns", "advantages"]
buf = ReplayBuffer(keys, buffer_size, mini_batch_size, device)

frames_seen, rollouts = 0, 0
start_time = time.time()

# --- Train loop ---
while frames_seen < max_frames:
    model.train()
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    logps, vals, states, actions, rewards, masks = deque(), deque(), deque(), deque(), deque(), deque()

    with torch.no_grad():
        for step in range(num_steps):
            dist, value = model(state)                       # [1, A], [1, 1]
            action = dist.sample()                           # [1, A] continuous
            # store tensors for training
            states.append(state)
            actions.append(action)
            vals.append(value)
            logps.append(dist.log_prob(action).sum(1, keepdim=True))  # [1,1]

            # step env with numpy action (flattened)
            env_action = action.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, r, done, info = env.step(env_action)
            if rollouts%50==0:
                env.render()
            rewards.append(torch.tensor([[r]], dtype=torch.float32, device=device))
            masks.append(torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=device))

            state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            frames_seen += 1

        # bootstrap
        _, next_v = model(state)
        rets, advs = compute_gae(next_v, rewards, masks, vals, gamma, lam)

        # normalize advantages
        advs_t = torch.cat(list(advs)).detach()
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        # fill buffer
        buf.data_log("states",   torch.cat(list(states)))
        buf.data_log("actions",  torch.cat(list(actions)))
        buf.data_log("values",   torch.cat(list(vals)).detach())
        buf.data_log("log_probs",torch.cat(list(logps)).detach())
        buf.data_log("returns",  torch.cat(list(rets)).detach())
        buf.data_log("advantages", advs_t.detach())

    # --- PPO updates ---
    for _ in range(ppo_epochs):
        for batch in buf:
            dist_new, v_new = model(batch["states"])
            entropy = dist_new.entropy().sum(dim=1).mean()   # sum over action dims
            loss_actor  = ppo_actor_loss(dist_new, batch["actions"], batch["log_probs"], batch["advantages"], clip_eps)
            loss_critic = ppo_critic_loss(v_new, batch["values"], batch["returns"], clip_eps)
            loss = loss_actor + loss_critic - 0.01 * entropy

            optim_.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            optim_.step()

    rollouts += 1
    if rollouts % 10 == 0:
        elapsed = int(time.time() - start_time)
        print(f"rollouts={rollouts} frames={frames_seen} time={elapsed}s")

env.close()
