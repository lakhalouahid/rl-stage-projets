import gym
import torch
import pdb

import numpy as np


from utils import Transform
from time import time
from torch import nn
from torch.optim import Adam, lr_scheduler



class carray():
  def __init__(self, maxlen, size, dtype, device):
    if isinstance(size, tuple):
      shape = (maxlen, *size)
    else:
      shape = (maxlen, size)
    self.buf = torch.ones(shape, dtype=dtype).to(device)
    self.maxlen = maxlen
    self.head = 0
    self.len = 0

  def append(self, item):
    self.buf[self.head:self.head+1] = item
    if self.len < self.maxlen:
      self.len += 1
    if self.head == 0:
      self.head = self.maxlen
    self.head -= 1

  def sample_from_perm(self, perm):
    perm = perm + self.head + 1
    perm = perm & (self.maxlen - 1)
    return self.buf[perm]


  def get_range_ilen(self, size, offset=0):
    rg = torch.arange(self.head + 1 + offset, self.head + 1 + offset + size)
    rg = rg & (self.maxlen - 1)
    return self.buf[rg]


class BufferRaw():
  def __init__(self, maxlen: int, device, state_shape, sequence_n):
    self.state_shape = state_shape
    self.states = carray(maxlen, state_shape, torch.float32, device)
    self.actions = carray(maxlen, 1, torch.int8, device)
    self.rewards = carray(maxlen, 1, torch.int8, device)
    self.tokens = carray(maxlen, 1, torch.int8, device)
    self.maxlen = maxlen
    self.device = device
    self.sequence_n = sequence_n
    self.len = 0

  def push(self, experience):
    state, action, reward, token = experience
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.tokens.append(token)
    if self.len < self.maxlen:
      self.len += 1

  def sample(self, batch_size):
    perm = np.random.permutation(self.len)[:batch_size]
    states = torch.ones(size=(perm.shape[0], self.sequence_n+1, *self.state_shape[-2:])).to(self.device)
    tokens = torch.ones(size=(perm.shape[0], self.sequence_n+1, 1))

    for i in range(self.sequence_n+1):
      states[:, i:i+1] = self.states.sample_from_perm(perm+i)
      tokens[:, i] = self.tokens.sample_from_perm(perm+i)

    sbixs, sfixs = np.where(tokens == -1)[:2]
    for i in range(sbixs.shape[0]):
      for j in range(sfixs[i]+1, self.sequence_n+1):
        states[sbixs[i], j] = states[sbixs[i], sfixs[i]]

    actions = self.actions.sample_from_perm(perm)
    rewards = self.rewards.sample_from_perm(perm)
    dones = torch.as_tensor(tokens[:, self.sequence_n] == 1).to(self.device)

    ns = np.where(tokens[:, self.sequence_n] != -1)[0]
    return (states[ns], actions[ns], rewards[ns], dones[ns])


class NN(nn.Module):
  def __init__(self, sequente_n):
    super(NN, self).__init__()
    self.cv =  nn.Sequential(
      nn.Conv2d(sequente_n, 32, kernel_size=8, stride=4), # 20
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=4, stride=2), # 9
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=3, stride=2), # 4
      nn.LeakyReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(2048, 512),
      nn.LeakyReLU(inplace=True),
      nn.Linear(512, 32),
      nn.LeakyReLU(inplace=True),
      nn.Linear(32, 2),
    )
    for cv in self.cv.modules():
      if isinstance(cv, nn.Conv2d):
        nn.init.kaiming_uniform_(cv.weight)
        if cv.bias != None:
          nn.init.zeros_(cv.bias)

    for fc in self.fc.modules():
      if isinstance(fc, nn.Linear):
        nn.init.xavier_normal_(fc.weight)
        nn.init.zeros_(fc.bias)

  def forward(self, x):
    x = self.cv(x)
    return self.fc(torch.flatten(x, start_dim=1))


class DQNN(nn.Module):
  def __init__(self, nntype, sequence_n, discount=0.9, lr=3e-4, lr_decay=0.999,
      min_lr=3e-6, itermax=60):
    super(DQNN, self).__init__()
    self.train_net = nntype(sequence_n)
    self.target_net = nntype(sequence_n)
    self.target_net.load_state_dict(self.train_net.state_dict())
    self.target_itermax = itermax
    self.save_freq = 0
    self.target_iter = 0
    self.discount = discount
    self.min_lr = min_lr
    self.optimizer = Adam(params=self.train_net.parameters(), lr=lr, weight_decay=0, amsgrad=False)
    self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_decay)
    self.sequence_n = sequence_n

  def forward(self, x):
    return self.train_net(x)

  def forward_target(self, x):
    with torch.no_grad():
      self.target_net.eval()
      return self.target_net(x)


  def train_dqn(self, batch, steps_nmbr):
    self.train_net.train()
    states, actions, rewards, dones = batch
    q_newstates = self.forward_target(states[:, :self.sequence_n])
    maxq_newstates = torch.max(q_newstates, dim=1).values.reshape(-1, 1)
    maxq_newstates = (1 - torch.as_tensor(dones, dtype=torch.int32)) *  maxq_newstates
    for param in self.train_net.parameters():
      param.grad = None
    q_states = self(states[:, 1:self.sequence_n+1])
    loss = torch.mean((self.discount * maxq_newstates + rewards - \
        torch.take_along_dim(q_states, torch.as_tensor(actions, dtype=torch.int64), dim=1))**2)
    loss.backward()
    self.optimizer.step()
    self.target_switch(steps_nmbr)
    if self.scheduler.get_lr()[0] > self.min_lr:
      self.scheduler.step()

  def eval_dqn(self, x):
    return int(torch.argmax(self.forward_target(x), dim=1))

  def target_switch(self, steps_nmbr):
    self.target_iter += 1
    if self.target_iter >= self.target_itermax:
      self.target_net.load_state_dict(self.train_net.state_dict())
      self.target_net.eval()
      self.target_iter = 1
      self.target_itermax = min(self.target_itermax + 1, 1<<10)
      self.save_freq += 1
      if self.save_freq & 255 == 0:
        torch.save(self.state_dict(), f"checkpoints/deep-raw-q-model-{steps_nmbr}-{int(time()):5d}.pt")

  def load_model(self, path):
    self.load_state_dict(torch.load(path))


def choose_action(actions_values, eps: float=0.05):
  if np.random.sample() < (1 - eps):
    return int(torch.argmax(actions_values, dim=1))
  else:
    return np.random.randint(2)

def train(lr, lr_decay, min_lr, itermax):
  try:
    eps = 1
    sequence_n = 4
    image_shape = (84, 84)
    device = torch.device("cuda")
    env = gym.make('CartPole-v1')
    transform = Transform(device, image_shape)
    buffer = BufferRaw(1<<17, device, (1, *image_shape), sequence_n)
    dqnn = DQNN(NN, sequence_n, lr=lr, lr_decay=lr_decay, min_lr=min_lr, itermax=itermax).to(device)
    for i_episode in range(int(1<<20)):
      env.reset()
      eps = max(eps*0.999, 0.1)
      states = carray(sequence_n+1, image_shape, dtype=torch.float32, device=device)
      new_state = transform.prepare(env.render(mode='rgb_array'))
      buffer.push((new_state, 0, 0, -1))
      for _ in range(sequence_n):
        states.append(new_state)
      for t in range(500):
        if i_episode & 255 == 0:
          env.render()
          action = dqnn.eval_dqn(states.get_range_ilen(sequence_n)[None, ...])
        else:
          dqnn.train_net.eval()
          actions_values = dqnn(states.get_range_ilen(sequence_n)[None, ...])
          action = choose_action(actions_values, eps)
        next_position, reward, done, _ = env.step(action)
        if abs(next_position[0]) > 3.2:
          done = True
        new_state = transform.prepare(env.render(mode='rgb_array'))
        states.append(new_state)
        buffer.push((new_state, action, reward, done*1))
        if done:
          if i_episode & 1 == 0:
            dqnn.train_dqn(buffer.sample(1<<8), t+1)
          if i_episode & 127 == 0:
            print(f"Test: episode {i_episode} finished after {t+1} steps")
          else:
            print(f"Episode {i_episode} finished after {t+1} steps")
          break

    env.close()
  except KeyboardInterrupt as _:
    print("exit")

def main():
  train(3e-3, 0.9995, 2.5e-4, 100)

if __name__  == "__main__":
  main()
