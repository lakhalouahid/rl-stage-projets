import gym
import torch
import math


import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T


from time import time
from PIL import Image
from torch import nn
from torch.optim import Adam, lr_scheduler





class Transform:
  def __init__(self, device, image_shape):
    self.device = device
    self.to_PILImage = T.ToPILImage()
    self.resize = T.Resize(image_shape, interpolation=Image.CUBIC)
    self.to_grayscale = T.Grayscale()
    self.from_PILImage = T.ToTensor()

  def prepare(self, x):
    x = x[40:340, 150:450]
    x = self.to_PILImage(x)
    x = self.resize(x)
    x = self.to_grayscale(x)
    x = self.from_PILImage(x)
    return x.to(self.device)



class carray():
  def __init__(self, maxlen, size, dtype, device):
    if isinstance(size, tuple):
      shape = (maxlen, *size)
    else:
      shape = (maxlen, size)
    self.buf = torch.zeros(shape, dtype=dtype).to(device)
    self.maxlen = maxlen
    self.head = 0
    self.len = 0

  def append(self, item):
    self.buf[self.head] = item
    if self.len < self.maxlen:
      self.len += 1
    if self.head == 0:
      self.head = self.maxlen
    self.head -= 1

  def sample(self, n):
    perm = torch.randperm(self.len)[:min(self.len, n)] + self.head + 1
    perm = perm & (self.maxlen - 1)
    return self.buf[perm]

  def sample_from_perm(self, perm):
    perm = perm + self.head + 1
    perm = perm & (self.maxlen - 1)
    return self.buf[perm]



class BufferRaw():
  def __init__(self, maxlen: int, device, state_shape):
    self.states = carray(maxlen, state_shape, torch.float32, device)
    self.actions = carray(maxlen, 1, torch.int8, device)
    self.newstates = carray(maxlen, state_shape, torch.float32, device)
    self.rewards = carray(maxlen, 1, torch.int8, device)
    self.dones = carray(maxlen, 1, torch.bool, device)
    self.maxlen = maxlen
    self.len = 0

  def push(self, experience):
    state, action, newstate, reward, done = experience
    self.states.append(state)
    self.actions.append(action)
    self.newstates.append(newstate)
    self.rewards.append(reward)
    self.dones.append(done)
    if self.len < self.maxlen:
      self.len += 1

  def sample(self, batch_size):
    perm = np.random.permutation(self.len)[:batch_size]
    states = self.states.sample_from_perm(perm)
    actions = self.actions.sample_from_perm(perm)
    newstates = self.newstates.sample_from_perm(perm)
    rewards = self.rewards.sample_from_perm(perm)
    dones = self.dones.sample_from_perm(perm)
    return (states, actions, newstates, rewards, dones)


class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.cv =  nn.Sequential(
      nn.Conv2d(1, 2, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(2),
      nn.ReLU(inplace=True),
      nn.Conv2d(2, 4, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(4),
      nn.ReLU(inplace=True),
      nn.Conv2d(4, 8, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU(inplace=True),
      nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU(inplace=True),
      nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU(inplace=True),
      nn.Conv2d(8, 8, kernel_size=2, stride=2, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU(inplace=True),
    )

    self.fc = nn.Sequential(
      nn.Linear(32, 8),
      nn.ReLU(inplace=True),
      nn.Linear(8, 8),
      nn.ReLU(inplace=True),
      nn.Linear(8, 2),
    )
    for cv in self.cv.modules():
      if isinstance(cv, nn.Conv2d):
        nn.init.kaiming_uniform_(cv.weight)
    for fc in self.fc.modules():
      if isinstance(fc, nn.Linear):
        nn.init.xavier_normal_(fc.weight)
        nn.init.zeros_(fc.bias)

  def forward(self, x):
    x = self.cv(x)
    return self.fc(torch.flatten(x, start_dim=1))


class DQNN(nn.Module):
  def __init__(self, nntype, discount=1, lr=3e-4, lr_decay=0.999, min_lr=3e-6, itermax=60):
    super(DQNN, self).__init__()
    self.train_net = nntype()
    self.target_net = nntype()
    self.target_itermax = itermax
    self.save_freq = 0
    self.target_iter = 0
    self.discount = discount
    self.min_lr = min_lr
    self.optimizer = Adam(params=self.train_net.parameters(), lr=lr, weight_decay=0, amsgrad=False)
    self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_decay)

  def forward(self, x):
    return self.train_net(x)

  def forward_target(self, x):
    with torch.no_grad():
      return self.target_net(x)


  def train_dqn(self, batch, steps_nmbr):
    self.train_net.train()
    states, actions, newstates, rewards, dones = batch
    with torch.no_grad():
      q_newstates = self.forward_target(newstates)
    maxq_newstates = torch.max(q_newstates, dim=1).values.reshape(-1, 1)
    maxq_newstates = (1 - torch.as_tensor(dones, dtype=torch.int32)) *  maxq_newstates
    for param in self.parameters():
      param.grad = None
    q_states = self(states)
    loss = torch.mean((self.discount * maxq_newstates + rewards - \
        torch.take_along_dim(q_states, torch.as_tensor(actions, dtype=torch.int64), dim=1))**2)
    loss.backward()
    self.optimizer.step()
    self.target_switch(steps_nmbr)
    if self.scheduler.get_lr()[0] > self.min_lr:
      self.scheduler.step()

  def eval_dqn(self, x):
    self.train_net.eval()
    return int(torch.argmax(self.forward(x), dim=1))

  def target_switch(self, steps_nmbr):
    self.target_iter += 1
    if self.target_iter >= self.target_itermax:
      self.target_net.load_state_dict(self.train_net.state_dict())
      self.target_net.eval()
      self.target_iter = 1
      self.save_freq += 1
      if self.save_freq & 255 == 0:
        torch.save(self.state_dict(), f"checkpoints/deep-raw-q-model-{steps_nmbr}-{int(time()):5d}.pt")
  def load_model(self, path):
    self.load_state_dict(torch.load(path))


def choose_action(actions_values, eps: float=0.05):
  return int(torch.argmax(actions_values, dim=1)) \
      if np.random.sample() < (1 - eps) else np.random.randint(2)

def train(lr, lr_decay, min_lr, itermax):
  try:
    eps = 0.24
    image_shape = (159, 159)
    device = torch.device("cpu")
    env = gym.make('CartPole-v1')
    transform = Transform(device, image_shape)
    buffer = BufferRaw(1<<16, device, (1, *image_shape))
    dqnn = DQNN(NN, lr=lr, lr_decay=lr_decay, min_lr=min_lr, itermax=itermax).to(device)
    for i_episode in range(int(1<<20)):
      eps = max(eps*0.999, 0.08)
      env.reset()
      newstate = transform.prepare(env.render(mode='rgb_array'))
      for t in range(500):
        state = newstate
        action = 0
        if i_episode & 255 == 0:
          # env.render()
          action = dqnn.eval_dqn(state[None, ...])
        else:
          actions_values = dqnn(state[None, ...])
          action = choose_action(actions_values, eps)
        next_position, reward, done, _ = env.step(action)
        if abs(next_position[0]) > 2.4:
          done = True
        newstate = transform.prepare(env.render(mode='rgb_array'))
        buffer.push((state, action, newstate, reward, done))
        state = newstate
        if done:
          if i_episode & 3 == 0:
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
  train(3e-3, 0.9999, 1e-3, 256)

if __name__  == "__main__":
  main()
