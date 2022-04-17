import gym
import torch
import math
import multiprocessing

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T


from time import time
from PIL import Image
from torch import nn
from torch.optim import Adam, lr_scheduler

class Transform:
  def __init__(self, device):
    self.device = device
    self.resize = T.Compose([
                    T.ToPILImage(),
                    T.Resize((40, 40), interpolation=Image.CUBIC),
                    T.ToTensor()
                  ])

  def prepare(self, x):
    x = x.transpose((2, 0, 1))
    x =  self.resize(x) / 255
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
  def __init__(self, maxlen: int, device):
    self.states = carray(maxlen, (3, 40, 40), torch.float32, device)
    self.actions = carray(maxlen, 1, torch.int8, device)
    self.newstates = carray(maxlen, (3, 40, 40), torch.float32, device)
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
  def __init__(self, h, w, outputs):
    super(NN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)

    convw = NN.conv2d_size_out(NN.conv2d_size_out(NN.conv2d_size_out(w)))
    convh = NN.conv2d_size_out(NN.conv2d_size_out(NN.conv2d_size_out(h)))
    linear_input_size = convw * convh * 32
    self.head = nn.Linear(linear_input_size, outputs)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    return self.head(x.view(x.size(0), -1))

  @staticmethod
  def conv2d_size_out(size, kernel_size = 5, stride = 2):
    return (size - (kernel_size - 1) - 1) // stride  + 1

class DQNN(nn.Module):
  def __init__(self, nntype, discount=1, lr=3e-4, lr_decay=0.999, min_lr=3e-6, itermax=60):
    super(DQNN, self).__init__()
    self.train_net = nntype(40, 40, 2)
    self.target_net = nntype(40, 40, 2)
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
    return self.target_net(x)


  def train_dqn(self, batch, steps_nmbr):
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
    with torch.no_grad():
      return int(torch.argmax(self.forward_target(x[None, ...]), dim=1))

  def target_switch(self, steps_nmbr):
    self.target_iter += 1
    if self.target_iter >= self.target_itermax:
      self.target_net.load_state_dict(self.train_net.state_dict())
      self.target_iter = 0
      self.save_freq += 1
      if self.save_freq & 3 == 0:
        torch.save(self.state_dict(), f"checkpoints/deep-raw-q-model-{steps_nmbr}-{int(time()):5d}.pt")
  def load_model(self, path):
    self.load_state_dict(torch.load(path))


def choose_action(actions_values, eps: float=0.05):
  return int(torch.argmax(actions_values, dim=1)) \
      if np.random.sample() < (1 - eps) else np.random.randint(2)

def train(lr, lr_decay, min_lr, itermax):
  try:
    eps = 0.2
    device = torch.device("cuda")
    env = gym.make('CartPole-v1', )
    transform = Transform(device)
    buffer = BufferRaw(1<<16, device)
    eps = max(eps*0.999, 0.05)
    dqnn = DQNN(NN, lr=lr, lr_decay=lr_decay, min_lr=min_lr, itermax=itermax).to(device)
    for i_episode in range(int(1e5)):
      env.reset()
      newstate = transform.prepare(env.render(mode='rgb_array'))
      for t in range(500):
        state = newstate
        action = 0
        if i_episode & 255 == 0:
          env.render(mode='human')
          action = dqnn.eval_dqn(state)
        else:
          actions_values = dqnn(state[None, ...])
          action = choose_action(actions_values, eps)
        _, reward, done, _ = env.step(action)
        newstate = transform.prepare(env.render(mode='rgb_array'))
        buffer.push((state, action, newstate, reward, done))
        state = newstate
        if done:
          dqnn.train_dqn(buffer.sample(1<<4), t+1)
          if i_episode & 7 == 0:
            if i_episode & 255 == 0:
              print(f"Test episode {i_episode} finished after {t+1} steps")
            else:
              print(f"Episode {i_episode} finished after {t+1} steps")
          break

    env.close()
  except KeyboardInterrupt as _:
    print("exit")

def main():
  train(1e-3, 0.999, 1e-4, 32)

if __name__  == "__main__":
  main()
