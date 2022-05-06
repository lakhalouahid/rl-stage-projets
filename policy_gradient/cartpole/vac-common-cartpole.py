import gym
import torch
import scipy.signal
from torch import nn
import numpy as np


env = gym.make("CartPole-v1")
device = torch.device("cpu")
num_episodes = 1<<16
max_episode_len = 500
hidden_size = 256
action_size = 2
state_size = 4
discount = 0.95
action_space = [0, 1]
device  = torch.device("cpu")
T = 1



def discount_cumsum(x, discount):
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ActorCriticNetwork(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.ln1 = nn.Linear(state_size , hidden_size)
    self.ln2 = nn.Linear(hidden_size, hidden_size)
    self.val = nn.Linear(hidden_size, action_size)
    self.pol = nn.Linear(hidden_size, action_size)
    self.common = None

  def forward_common(self, x):
    x = torch.as_tensor(x, dtype=torch.float32).to(device)
    x = torch.relu(self.ln1(x))
    self.common = torch.relu(self.ln2(x))

  def forward_policy(self):
    x = self.pol(self.common)
    x = x + x.sum() * 5e-3
    actions = torch.softmax(x, dim=1)
    action = self.get_action(actions.squeeze(0))
    logprob_actions = torch.log(actions.squeeze(0))
    logprob_action = logprob_actions[action]
    return action, logprob_action

  def forward_value(self, action):
    x = self.val(self.common).squeeze(0)[action]
    return x

  def get_action(self, actions):
    return np.random.choice(action_space, p=actions.detach().numpy())


class Episode(object):
  def __init__(self):
    self.s, self.a, self.lp, self.q = [], [], [], []
    self.nx_s, self.nx_a, self.nx_lp, self.nx_q = [], [], [], []
    self.r = []
    self.d = []

  def push(self, s, a, lp, q, r, nx_s, nx_a, nx_lp, nx_q, d):
    self.s.append(s)
    self.a.append(a)
    self.q.append(q)
    self.lp.append(lp)
    self.nx_s.append(nx_s)
    self.nx_a.append(nx_a)
    self.nx_q.append(nx_q)
    self.nx_lp.append(nx_lp)
    self.r.append(r)
    self.d.append(d)

  def finish_episode(self):
    self.s_ = torch.tensor(np.array(self.s), dtype=torch.float32)
    self.a_ = torch.tensor(self.a, dtype=torch.float32)
    self.lp_ = torch.stack(self.lp)
    self.q_ = torch.stack(self.q)
    self.nx_s_ = torch.tensor(np.array(self.nx_s), dtype=torch.float32)
    self.nx_a_ = torch.tensor(self.nx_a, dtype=torch.float32)
    self.nx_q_ = torch.stack(self.nx_q)
    self.nx_lp_ = torch.stack(self.nx_lp)
    self.r_ = torch.tensor(np.array(self.r), dtype=torch.float32)
    self.d_ = torch.tensor(np.array(self.d), dtype=torch.float32)


ac_net = ActorCriticNetwork()
ac_optim = torch.optim.Adam(ac_net.parameters(), lr=0.0003)


def train(e):
  v = e.q_.detach()
  lp = e.lp_
  v = (v - torch.mean(v)) / torch.std(v)
  ac_net.zero_grad()
  policy_loss = -lp * v
  value_loss = (e.r_ + (1.0 - e.d_) * discount * e.nx_q_.detach() - e.q_)**2
  policy_loss.mean().backward(retain_graph=True)
  value_loss.mean().backward(retain_graph=True)
  ac_optim.step()
try:
  for i in range(num_episodes):
    nx_s = env.reset()
    ac_net.forward_common(nx_s[None, ...])
    nx_a, nx_lp = ac_net.forward_policy()
    nx_q = ac_net.forward_value(nx_a)
    episode = Episode()
    for j in range(max_episode_len):
      # env.render()
      s, a, q, lp = nx_s, nx_a, nx_q, nx_lp
      nx_s, r, d, _ = env.step(a)
      ac_net.forward_common(nx_s[None, ...])
      nx_a, nx_lp = ac_net.forward_policy()
      nx_q = ac_net.forward_value(nx_a)
      episode.push(s, a, lp, q, r, nx_s, nx_a, nx_lp, nx_q, d * 1.0)
      s = nx_s
      if d:
        print(f"Episode {i} finished after {j+1} steps")
        break
    if i & 15 == 0:
      episode.finish_episode()
      train(episode)
except KeyboardInterrupt:
  pass
