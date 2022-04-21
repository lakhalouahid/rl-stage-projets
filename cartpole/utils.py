from PIL import Image



import numpy as np
import torchvision.transforms as T


class carray():
  def __init__(self, maxlen, size, dtype):
    if isinstance(size, tuple):
      shape = (maxlen, *size)
    else:
      shape = (maxlen, size)
    self.buf = np.zeros(shape=shape, dtype=dtype)
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
    perm = np.random.permutation(self.len)[:min(self.len, n)] + self.head + 1
    perm = perm & (self.maxlen - 1)
    return self.buf[perm]

  def sample_from_perm(self, perm):
    perm = perm + self.head + 1
    perm = perm & (self.maxlen - 1)
    return self.buf[perm]



class Buffer():
  def __init__(self, maxlen: int):
    self.states = carray(maxlen, 4, np.float32)
    self.actions = carray(maxlen, 1, np.int8)
    self.newstates = carray(maxlen, 4, np.float32)
    self.rewards = carray(maxlen, 1, np.int8)
    self.dones = carray(maxlen, 1, np.bool8)
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



class Transform:
  def __init__(self, device, image_shape):
    self.device = device
    self.to_PILImage = T.ToPILImage()
    self.resize = T.Resize(image_shape, interpolation=Image.CUBIC)
    self.to_grayscale = T.Grayscale()
    self.from_PILImage = T.ToTensor()

  def prepare(self, x):
    x = x[0:400,100:500]
    x = self.to_PILImage(x)
    x = self.resize(x)
    x = self.to_grayscale(x)
    x = self.from_PILImage(x)
    return x.to(self.device)
