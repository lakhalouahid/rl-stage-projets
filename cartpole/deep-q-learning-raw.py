import gym
import torch
import math
from collections import deque

import numpy as np

from torch import nn
from torch.optim import Adam, lr_scheduler

import logging
logger = logging.basicConfig(filename="deep-double-q-learning.log", level=logging.INFO, format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s")


class Buffer():
  """
  Buffer implementation
  """
  def __init__(self, maxlen: int=100000):
    """
    Initialize new buffer with the maxlength defaulted to 100000
    """
    self._states = deque(maxlen=maxlen)
    self._actions = deque(maxlen=maxlen)
    self._newstates = deque(maxlen=maxlen)
    self._rewards = deque(maxlen=maxlen)
    self._dones = deque(maxlen=maxlen)
    self._length = deque(maxlen=maxlen)

  def push(self, experience):
    """
    Push new experience to the buffer
    """
    state, action, newstate, reward, done = experience
    self._states.append(state)
    self._actions.append(action)
    self._newstates.append(newstate)
    self._rewards.append(reward)
    self._dones.append(done)

  def sample(self, batch_size: int=8):
    """
    Sample from the buffer without replacement
    """
    batch_size = min(batch_size, len(self._states))
    states = np.zeros((batch_size, 4), dtype=np.float32)
    newstates = np.zeros((batch_size, 4), dtype=np.float32)
    actions = np.zeros((batch_size, 1), dtype=np.int64)
    rewards = np.zeros((batch_size, 1), dtype=np.float32)
    dones = np.zeros((batch_size, 1), dtype=np.bool8)
    for i in range(batch_size):
      idx = np.random.randint(len(self._states))
      states[i] = self._states[idx]
      newstates[i] = self._newstates[idx]
      actions[i] = self._actions[idx]
      rewards[i] = self._rewards[idx]
      dones[i] = self._dones[idx]
      del self._states[idx], self._newstates[idx], self._actions[idx], self._rewards[idx], self._dones[idx]
    return (states, actions, newstates, rewards, dones)

class NN(nn.Sequential):
  def __init__(self):
    super(NN, self).__init__(
        nn.Linear(4, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 2),
        )

class QNN(nn.Module):
  """
  Q learning neural network
  """
  def __init__(self, nntype, use_target=False, discount=1, lr=1e-3, min_lr=8e-5):
    super(QNN, self).__init__()
    self.layers = nntype()
    self.target = nntype()
    self.target_itermax = 60
    self.target_iter = 0
    self.use_target = use_target
    self.discount = discount
    self.lr = lr
    self.min_lr = min_lr
    for m in self.layers.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

    self.optimizer = Adam(params=self.parameters(), lr=self.lr, weight_decay=0, amsgrad=False)
    self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.999)

  def forward(self, x):
    x = torch.from_numpy(x)
    return self.layers(x)

  def forward_target(self, x):
    x = torch.from_numpy(x)
    return self.target(x)


  def train_qdn(self, batch):
    """
    Train the q learning nn using a batch
    """
    states, actions, newstates, rewards, dones = batch
    actions = torch.from_numpy(actions)
    rewards = torch.from_numpy(rewards)
    dones = torch.from_numpy(dones)
    with torch.no_grad():
      q_newstates =self.forward_target(newstates)
    maxq_newstates = torch.max(q_newstates, dim=1).values.reshape(-1, 1)
    maxq_newstates = (1 - torch.as_tensor(dones, dtype=torch.int32)) *  maxq_newstates
    for param in self.parameters():
      param.grad = None
    q_states = self(states)
    loss = self.loss_func(q_states, actions, maxq_newstates, rewards)
    loss.backward()
    self.optimizer.step()
    self.target_switch()
    if self.scheduler.get_lr()[0] > self.min_lr:
      self.scheduler.step()

  def target_switch(self):
    """
    Update the target network
    """
    self.target_iter += 1
    if self.target_iter >= self.target_itermax:
      self.target.load_state_dict(self.layers.state_dict())
      self.target_iter = 0
      torch.save(self.state_dict(), "deep-q-model.pt")

  def loss_func(self, q_states, actions, maxq_newstates, rewards):
    """
    Loss function for q learning nn
    """
    loss = torch.mean((self.discount * maxq_newstates + rewards - torch.take_along_dim(q_states, actions, dim=1))**2)
    return loss

def choose_action(actions_values, eps: float=0.05):
  """
  Choose the action using epsilon-greedy policy
  """
  return int(torch.argmax(actions_values)) \
      if (np.random.sample() < (1 - eps)) \
      else np.random.randint(actions_values.shape[0])


def main():
  try:
    eps = 0.2
    env = gym.make('CartPole-v1')
    buffer = Buffer()
    qnn = QNN(NN)
    for i_episode in range(int(1e5)):
      state = env.reset()
      eps = max(eps * 0.995, 0.01)
      for t in range(1000):
        # env.render()
        actions_values = qnn(state)
        action = choose_action(actions_values, eps)
        newstate, reward, done, info = env.step(action)
        buffer.push((state, action, newstate, reward, done))
        state = newstate
        if done:
          qnn.train_qdn(buffer.sample(1000))
          if i_episode & 7 == 0:
            #print(f"Episode {i_episode} finished after {t+1} steps")
            logging.info(f"Episode {i_episode} finished after {t+1} steps")
          break

    env.close()
  except KeyboardInterrupt as e:
    print("exit")

if __name__  == "__main__":
  main2()
