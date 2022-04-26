import gym
import time
import torch
import logging

import numpy as np


from utils import Transform
from torch import nn
from torch.optim import Adam, lr_scheduler


device = torch.device("cuda")
logging.basicConfig(filename=f"./logs/q-learning-cnn-int({time.time()})", format="%(message)s", level=logging.DEBUG)

def choose_action(actions_values, eps: float=0.05):
  if np.random.sample() < (1 - eps):
    return int(torch.argmax(actions_values, dim=1))
  else:
    return np.random.randint(2)

class carray():
  def __init__(self, maxlen, size, dtype, device):
    if isinstance(size, tuple):
      shape = (maxlen, *size)
    else:
      shape = (maxlen, size)
    self.buffer = torch.zeros(shape, dtype=dtype).to(device)
    self.maxlen = maxlen
    self.head = maxlen - 1

  def append(self, item):
    self.buffer[self.head:self.head+1] = item
    if self.head == 0:
      self.head = self.maxlen
    self.head -= 1

  def sample_from_perm(self, perm):
    perm = perm + self.head + 1
    perm = perm & (self.maxlen - 1)
    return self.buffer[perm]


  def get_range_ilen(self, size, offset=0):
    rg = torch.arange(offset, offset + size) + self.head + 1
    rg = rg & (self.maxlen - 1)
    return self.buffer[rg]



class BufferRaw():
  def __init__(self, maxlen):
    self.states = carray(maxlen, (2, 84, 84), torch.uint8, device)
    self.new_states = carray(maxlen, (2, 84, 84), torch.uint8, device)
    self.actions = carray(maxlen, 1, torch.int8, device)
    self.rewards = carray(maxlen, 1, torch.int8, device)
    self.dones = carray(maxlen, 1, torch.int8, device)
    self.maxlen = maxlen
    self.len = 0

  def push(self, experience):
    state, action, new_state, reward, done = experience
    self.states.append(state)
    self.actions.append(action)
    self.new_states.append(new_state)
    self.rewards.append(reward)
    self.dones.append(done)
    if self.len < self.maxlen:
      self.len += 1

  def sample(self, batch_size):
    perm = np.random.permutation(self.len)[:batch_size]
    return (
      self.states.sample_from_perm(perm),
      self.actions.sample_from_perm(perm),
      self.new_states.sample_from_perm(perm),
      self.rewards.sample_from_perm(perm),
      self.dones.sample_from_perm(perm),
      )

class NN(nn.Module):
  def __init__(self, sequente_n):
    super(NN, self).__init__()
    self.cv =  nn.Sequential(
      nn.Conv2d(sequente_n, 32, kernel_size=8, stride=4),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=3, stride=2),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(2048, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 32),
      nn.ReLU(inplace=True),
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

class DQN(nn.Module):
  def __init__(self, nntype, discount=0.9, lr=3e-3, itermax=1000, lr_decay=0.9995, lr_min=8e-5):
    super(DQN, self).__init__()
    self.train_net = nntype(sequence_n)
    self.target_net = nntype(sequence_n)
    self.target_net.load_state_dict(self.train_net.state_dict())
    self.target_itermax = itermax
    self.target_iter = 0
    self.save_freq = 0
    self.lr_min = lr_min
    self.discount = discount
    self.optimizer = Adam(params=self.train_net.parameters(), lr=lr, weight_decay=0, amsgrad=False)
    self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_decay)

  def forward(self, x):
    x = torch.as_tensor(x, dtype=torch.float32) / 255.0
    return self.train_net(x)

  def forward_target(self, x):
    with torch.no_grad():
      x = torch.as_tensor(x, dtype=torch.float32) / 255.0
      return self.target_net(x)


  def train_dqn(self, batch):
    self.train_net.train()
    states, actions, new_states, rewards, dones = batch
    q_newstates = self.forward_target(new_states)
    maxq_newstates = torch.max(q_newstates, dim=1).values.reshape(-1, 1)
    maxq_newstates = (1 - torch.as_tensor(dones, dtype=torch.int32)) *  maxq_newstates
    self.optimizer.zero_grad()
    q_states = self(states)
    loss = torch.mean((self.discount * maxq_newstates + rewards - torch.take_along_dim(q_states, torch.as_tensor(actions, dtype=torch.int64), dim=1))**2)
    loss.backward()
    self.optimizer.step()
    self.target_switch()
    if self.scheduler.get_lr()[0] > self.lr_min:
      self.scheduler.step()

  def eval_dqn(self, x):
    with torch.no_grad():
      return int(torch.argmax(self.forward(x), dim=1))

  def target_switch(self):
    self.target_iter += 1
    if self.target_iter >= self.target_itermax:
      self.target_net.load_state_dict(self.train_net.state_dict())
      self.target_net.eval()
      self.target_iter = 1
      self.save_freq += 1
      torch.save(self.state_dict(), f"checkpoints/deep-q-model-{str(time.time())[-5:-1]}.pt")

  def load_model(self, path):
    self.load_state_dict(torch.load(path))



eps = 1
sequence_n = 2
image_shape = (84, 84)
device = torch.device("cuda")
env = gym.make('CartPole-v1')
transform = Transform(device, image_shape)
buffer = BufferRaw(1<<15)
dqnn = DQN(NN, discount=0.9, lr=1e-3, itermax=250, lr_min=8e-5, lr_decay=0.9999).to(device)
for i_episode in range(1<<20):
  env.reset()
  eps = 0.1 + (eps - 0.1) * 0.9995
  new_screen = transform.prepare(env.render(mode='rgb_array'))
  screen = new_screen
  new_state = torch.concat((new_screen, screen), dim=0)
  dqnn.train_net.eval()
  for t in range(500):
    state = new_state
    if i_episode & 255 == 0:
      env.render()
      action = dqnn.eval_dqn(state[None, ...])
    else:
      actions_values = dqnn(state[None, ...])
      action = choose_action(actions_values, eps)
    new_pos, reward, done, _ = env.step(action)
    if abs(new_pos[0]) >= 1.6:
      done = True
    screen = new_screen
    new_screen = transform.prepare(env.render(mode='rgb_array'))
    new_state = torch.concat((new_screen, screen), dim=0)
    buffer.push((state, action, new_state, reward, done*1))
    if done:
      dqnn.train_dqn(buffer.sample(1<<8))
      if i_episode & 127 == 0:
        print(f"Test: episode {i_episode} finished after {t+1} steps, eps = {eps}")
        logging.info(f"Test: episode {i_episode} finished after {t+1} steps, eps = {eps}")
      else:
        print(f"Episode {i_episode} finished after {t+1} steps, eps = {eps}")
        logging.info(f"Episode {i_episode} finished after {t+1} steps, eps = {eps}")
      break

env.close()
