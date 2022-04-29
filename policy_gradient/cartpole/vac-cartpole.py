import gym
import torch
import scipy.signal
from torch import nn
import numpy as np


env = gym.make("CartPole-v1")
device = torch.device("cpu")
num_episodes = 1<<10
max_episode_len = 500
hidden_size = 256
action_size = 2
state_size = 4
discount = 0.95
action_space = [0, 1]
device  = torch.device("cpu")



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
    self.next_s, self.next_a, self.next_lp, self.next_q = [], [], [], []
    self.r = []

  def push(self, s, a, lp, q, r, next_s, next_a, next_lp, next_q):
    self.s.append(s)
    self.a.append(a)
    self.q.append(q)
    self.lp.append(lp)
    self.next_s.append(next_s)
    self.next_a.append(next_a)
    self.next_q.append(next_q)
    self.next_lp.append(next_lp)
    self.r.append(r)

  def finish_episode(self):
    self.s_ = torch.tensor(np.array(self.s), dtype=torch.float32)
    self.a_ = torch.tensor(self.a, dtype=torch.float32)
    self.lp_ = torch.stack(self.lp)
    self.q_ = torch.stack(self.q)
    self.next_s_ = torch.tensor(np.array(self.next_s), dtype=torch.float32)
    self.next_a_ = torch.tensor(self.next_a, dtype=torch.float32)
    self.next_lp_ = torch.stack(self.next_lp)
    self.next_q_ = torch.stack(self.next_q)
    self.r_ = torch.tensor(np.array(self.r), dtype=torch.float32)


policy_net = PolicyNetwork()
policy_opt = torch.optim.Adam(policy_net.parameters(), lr=0.001)

value_net = ValueNetwork()
value_opt = torch.optim.Adam(value_net.parameters(), lr=0.0001)

def train(episode):
  v = episode.q_.detach()
  lp = episode.lp_
  v = (v - torch.mean(v)) / torch.std(v)
  policy_loss = -lp * v
  policy_net.zero_grad()
  policy_loss.mean().backward()
  policy_opt.step()

  value_loss = (episode.r_ + discount * episode.next_q_.detach() - episode.q_)**2
  value_net.zero_grad()
  value_loss.mean().backward()
  value_opt.step()
try:
  for i in range(num_episodes):
    next_s = env.reset()
    next_a, next_lp = policy_net(next_s[None, ...])
    next_q = value_net(next_s[None, ...])[0][next_a]
    episode = Episode()
    for j in range(max_episode_len):
      # env.render()
      a, lp = next_a, next_lp
      s = next_s
      q = next_q
      next_s, r, d, _ = env.step(a)
      next_a, next_lp = policy_net(next_s[None, ...])
      next_q = value_net(next_s[None, ...])[0][next_a]
      episode.push(s, a, lp, q, r, next_s, next_a, next_lp, next_q)
      s = next_s
      if d:
        episode.finish_episode()
        train(episode)
        print(f"Episode finished after {j+1} steps")
        break
except KeyboardInterrupt:
  pass
