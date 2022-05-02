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

class ValueNetwork(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.ln1 = nn.Linear(state_size , hidden_size)
    self.ln2 = nn.Linear(hidden_size, hidden_size)
    self.ln3 = nn.Linear(hidden_size, action_size)
    nn.init.xavier_uniform_(self.ln1.weight)
    nn.init.zeros_(self.ln1.bias)
    nn.init.xavier_uniform_(self.ln2.weight)
    nn.init.zeros_(self.ln2.bias)
    nn.init.xavier_uniform_(self.ln3.weight)
    nn.init.zeros_(self.ln3.bias)

  def forward(self, x):
    x = torch.as_tensor(x, dtype=torch.float32).to(device)
    x = torch.relu(self.ln1(x))
    x = torch.relu(self.ln2(x))
    x = self.ln3(x)
    return x

class PolicyNetwork(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.ln1 = nn.Linear(state_size , hidden_size)
    self.ln2 = nn.Linear(hidden_size, hidden_size)
    self.ln3 = nn.Linear(hidden_size, action_size)
    nn.init.xavier_uniform_(self.ln1.weight)
    nn.init.zeros_(self.ln1.bias)
    nn.init.xavier_uniform_(self.ln2.weight)
    nn.init.zeros_(self.ln2.bias)
    nn.init.xavier_uniform_(self.ln3.weight)
    nn.init.zeros_(self.ln3.bias)

  def forward(self, x):
    x = torch.as_tensor(x, dtype=torch.float32).to(device)
    x = torch.relu(self.ln1(x))
    x = torch.relu(self.ln2(x))
    x = self.ln3(x)
    x = x + x.sum() * 5e-3
    actions = torch.softmax(x, dim=1)
    action = self.get_action(actions.squeeze(0))
    logprob_actions = torch.log(actions.squeeze(0))
    logprob_action = logprob_actions[action]
    return action, logprob_action

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


policy_net = PolicyNetwork()
policy_opt = torch.optim.Adam(policy_net.parameters(), lr=0.001)

value_net = ValueNetwork()
value_opt = torch.optim.Adam(value_net.parameters(), lr=0.0001)

def train(e):
  v = e.q_.detach()
  lp = e.lp_
  v = (v - torch.mean(v)) / torch.std(v)
  policy_loss = -lp * v
  policy_net.zero_grad()
  policy_loss.mean().backward()
  policy_opt.step()

  value_loss = (e.r_ + (1.0 - e.d_) * discount * e.nx_q_ - e.q_)**2
  value_net.zero_grad()
  value_loss.mean().backward()
  value_opt.step()
try:
  for i in range(num_episodes):
    nx_s = env.reset()
    nx_a, nx_lp = policy_net(nx_s[None, ...])
    nx_q = value_net(nx_s[None, ...])[0][nx_a]
    episode = Episode()
    for j in range(max_episode_len):
      # env.render()
      s, a, q, lp = nx_s, nx_a, nx_q, nx_lp
      nx_s, r, d, _ = env.step(a)
      nx_a, nx_lp = policy_net(nx_s[None, ...])
      with torch.no_grad():
        nx_q = value_net(nx_s[None, ...])[0][nx_a]
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
