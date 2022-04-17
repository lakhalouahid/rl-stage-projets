import gym
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam, lr_scheduler


import logging

logger = logging.basicConfig(filename="deep-double-q-learning.log", level=logging.INFO, format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s")




class Buffer():
  """
  Buffer implementation
  """
  def __init__(self, maxlength: int=100000):
    """
    Initialize new buffer with the maxlength defaulted to 100000
    """
    self._actions = []
    self._states = []
    self._newstates = []
    self._rewards = []
    self._dones = []
    self._qindexes = []
    self._length = maxlength

  def push(self, experience):
    """
    Push new experience to the buffer
    """
    if len(self._states) == self._length:
      self._states.pop()
      self._actions.pop()
      self._newstates.pop()
      self._rewards.pop()
      self._dones.pop()
      self._qindexes.pop()

    state, action, newstate, reward, done, qindex = experience
    self._states.insert(0, state)
    self._actions.insert(0, action)
    self._newstates.insert(0, newstate)
    self._rewards.insert(0, reward)
    self._dones.insert(0, done)
    self._qindexes.insert(0, qindex)

  def sample(self, minibatch_size: int=8):
    """
    Sample from the buffer without replacement
    """
    batch_size = min(minibatch_size, len(self._states))
    states = np.zeros((batch_size, 4), dtype=np.float32)
    newstates = np.zeros((batch_size, 4), dtype=np.float32)
    actions = np.zeros((batch_size, 1), dtype=np.int64)
    rewards = np.zeros((batch_size, 1), dtype=np.float32)
    dones = np.zeros((batch_size, 1), dtype=np.bool8)
    qindexes = np.zeros((batch_size, 1), dtype=np.bool8)
    for i in range(batch_size):
      idx = np.random.randint(len(self._states))
      states[i] = self._states.pop(idx)
      newstates[i] = self._newstates.pop(idx)
      actions[i] = self._actions.pop(idx)
      rewards[i] = self._rewards.pop(idx)
      dones[i] = self._dones.pop(idx)
      qindexes[i] = self._qindexes.pop(idx)
    return (states, actions, newstates, rewards, dones, qindexes)



class NN(nn.Sequential):
  def __init__(self):
    super(NN, self).__init__(
        nn.Linear(4, 32),
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
  def __init__(self, nntype, use_target=False, discount=1, lr=1e-3, min_lr=1e-5):
    self.discount = lr
    self.min_lr = min_lr
    super(QNN, self).__init__()
    self.layers = nntype()
    self.target = nntype()
    self.target_itermax = 50
    self.target_iter = 0
    self.use_target = use_target
    self.discount = discount
    self.lr = lr
    self.min_lr = min_lr
    self.optimizer = Adam(params=self.parameters(), lr=self.lr, weight_decay=0, amsgrad=False)
    self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9999)

    for m in self.layers.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

  def forward(self, x):
    x = torch.from_numpy(x)
    return self.layers(x)

  def forward_target(self, x):
    x = torch.from_numpy(x)
    return self.target(x)

  def loss_func(self, q_state, action, maxq_newstate, reward):
    """
    Loss function for q learning nn
    """
    # NOTE: the function needs batch of experiences
    loss = torch.mean((self.discount * maxq_newstate + reward - torch.take_along_dim(q_state, action, dim=1))**2)
    return loss

  @staticmethod
  def train_qdn(qnns: list, batch):
    """
    Train the two q-learning networks using a batch
    """
    states, actions, newstates, rewards, dones, qindexes = batch
    actions = torch.from_numpy(actions)
    rewards = torch.from_numpy(rewards)
    dones = torch.from_numpy(dones)
    qindexes = torch.from_numpy(qindexes)
    for i in range(states.shape[0]):
      qnn_this = qnns[qindexes[i]]
      qnn_other = qnns[int(qindexes[i])-1]
      with torch.no_grad():
        q_newstate = qnn_this.forward_target(newstates[i])
        maxaction = int(torch.argmax(q_newstate))
        maxq_newstate = qnn_other(newstates[i])[maxaction]
      for param in qnn_this.parameters():
        param.grad = None
      q_state = qnn_this(states[i])
      loss = (qnn_this.discount * maxq_newstate * float(1 - bool(dones[i])) + rewards[i] - q_state[actions[i]])**2
      loss.backward()
      qnn_this.optimizer.step()
      if qnn_this.scheduler.get_lr()[0] > qnn_this.min_lr:
        qnn_this.scheduler.step()
      qnn_this.target_switch()

  def target_switch(self):
    """
    Update the target network
    """
    self.target_iter += 1
    if self.target_iter >= self.target_itermax:
      self.target.load_state_dict(self.layers.state_dict())
      self.target_iter = 0
      torch.save(self.state_dict(), "deep-double-q-model.pt")

def choose_action(actions_values, eps: float=0.05):
  """
  Choose the action using epsilon-greedy policy
  """
  return int(torch.argmax(actions_values)) \
      if (np.random.sample() < (1 - eps)) \
      else np.random.randint(actions_values.shape[0])




def dblqnn_choose(qnns, state):
  """
  Choose randomly one qnn to predict the actions values
  """
  rand_idx = np.random.random().__round__()
  return qnns[rand_idx](state), rand_idx






def main():
  try:
    eps = 0.1
    env = gym.make('CartPole-v1')
    buffer = Buffer()
    qnns = [QNN(NN), QNN(NN)]
    for i_episode in range(20000):
      state = env.reset()
      eps = max(eps * 0.995, 0.001)
      for t in range(1000):
        # env.render()
        actions_values, qnn_idx = dblqnn_choose(qnns, state)
        action = choose_action(actions_values, eps)
        newstate, reward, done, info = env.step(action)
        buffer.push((state, action, newstate, reward, done, qnn_idx))
        state = newstate
        if done:
          QNN.train_qdn(qnns, buffer.sample(1000))
          if i_episode & 7 == 0:
            print(f"Episode {i_episode} finished after {t+1} steps")
            logging.info(f"Episode {i_episode} finished after {t+1} steps")
          break
    env.close()
  except KeyboardInterrupt as e:
    print("exit")


if __name__ == '__main__':
  main()
