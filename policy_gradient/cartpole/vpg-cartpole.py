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
    self.s = []
    self.r = []
    self.a = []
    self.lp = []

  def push(self, s, a, r, lp):
    self.s.append(s)
    self.a.append(a)
    self.r.append(r)
    self.lp.append(lp)

  def finish_episode(self):
    self.v = discount_cumsum(self.r, discount=discount)
    self.s_ = torch.tensor(np.array(self.s), dtype=torch.float32)
    self.v_ = torch.tensor(np.copy(self.v), dtype=torch.float32)
    self.a_ = torch.tensor(self.a, dtype=torch.float32)
    self.lp_ = torch.stack(self.lp)

policy_net = PolicyNetwork()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

def train(episode):
  v = episode.v_
  lp = episode.lp_
  v = (v - torch.mean(v)) / torch.std(v)
  policy_grad = -lp * v
  policy_net.zero_grad()
  policy_grad.mean().backward()
  optimizer.step()

try:
  for i in range(num_episodes):
    s = env.reset()
    episode = Episode()
    for j in range(max_episode_len):
      # env.render()
      a, lp = policy_net(s[None, ...])
      next_s, r, d, _ = env.step(a)
      episode.push(s, a, r, lp)
      s = next_s
      if d:
        episode.finish_episode()
        train(episode)
        print(f"Episode finished after {j+1} steps")
        break
except KeyboardInterrupt:
  pass
