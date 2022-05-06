from math import ceil
import os
import pdb
import torch
import pygame
import logging
import numpy as np



rom time import time
from torch import nn
from torchvision import transforms

device = torch.device("cuda:0")
logging.basicConfig(filename=f"logs/plain-autoencoder-{time():.0f}.log", format="%(message)s", level=logging.INFO)

w, h, n = 360, 360, 6
a, b = w//n, h//n
aa, bb = a-8, b-8



bg_color = (64, 64, 64)
action_space = (0, 1, 2, 3)
buffer_size = 1000
batch_size = 10


circ_center = (0, 0)
circ_radius = min(aa//2, bb//2)

rect_center = (0, 0)
rect_a, rect_b = aa, bb

poly_xy = [
  (0, +a//2),
  (-b//3, a//3),
  (-b//2, 0),
  (-b//3, -a//3),
  (0, -a//2),
  (b//3, -a//3),
  (+b//2, 0),
  (b//3, a//3),
]

colors = (
  (200, 00, 00),
  (00, 200, 00),
  (00, 00, 200),
  (200, 200, 0),
  (0, 200, 200),
  (200, 0, 200),
  (100, 0, 160),
)

moves = (
  (-1, 0),
  (0, -1),
  (+1, 0),
  (0, +1),
)

n_default = 8
color_default = 0
n_samples = 1000

class GridWorld():

  def __init__(self):
    self.grid = np.zeros((n, n), dtype=np.uint8)
    self.screen = pygame.Surface((w, h))
    self.screen.fill(bg_color)
    self.b = np.zeros((n, n), dtype=np.float32)
    self.n = np.ones((n, n), dtype=np.float32)

  def update_b(self, r):
    self.b[self.idx_a, self.idx_b] += r / self.n[self.idx_a, self.idx_b]
    self.n[self.idx_a, self.idx_b] = min(1.0 + self.n[self.idx_a, self.idx_b], 100)

  def get_b(self):
    return self.b[self.idx_a, self.idx_b]

  def initialize(self):
    for _ in range(n_default):
      color = colors[np.random.randint(1, len(colors))]
      idxs_a, idxs_b = np.where(self.grid == 0)
      idx = np.random.randint(len(idxs_a))
      idx_a, idx_b = idxs_a[idx], idxs_b[idx]
      self.grid[idx_a, idx_b] = 1
      center = GridWorld.compute_center(idx_a, idx_b)
      object_shape = np.random.randint(3)
      if object_shape == 0:
        pygame.draw.polygon(self.screen, color, GridWorld.abs_coords_pts(center, poly_xy))
      elif object_shape == 1:
        rect = pygame.rect.Rect((center[0] - rect_a//2, center[1] - rect_b//2), (rect_a, rect_b))
        pygame.draw.rect(self.screen, color, rect)
      else:
        pygame.draw.circle(self.screen, color, center, circ_radius)

    color = colors[color_default]
    idxs_a, idxs_b = np.where(self.grid == 0)
    idx = np.random.randint(len(idxs_a))
    self.idx_a, self.idx_b = idxs_a[idx], idxs_b[idx]
    center = GridWorld.compute_center(self.idx_a, self.idx_b)
    pygame.draw.circle(self.screen, color, center, circ_radius)
    self.grid[self.idx_a, self.idx_b] = 1

  def get_state(self):
    frame = pygame.surfarray.pixels3d(self.screen)
    pos = (self.idx_a, self.idx_b)
    return GridWorld.to_float_tensor(frame), pos

  def step(self, action):
    color = colors[color_default]
    pos = (self.idx_a, self.idx_b)
    next_pos = GridWorld.move(pos, action)
    if self.grid[next_pos] == 0:
      center = GridWorld.compute_center(*pos)
      rect = pygame.rect.Rect((center[0] - a//2, center[1] - b//2), (a, b))
      pygame.draw.rect(self.screen, bg_color, rect)
      self.grid[pos] = 0
      center = GridWorld.compute_center(*next_pos)
      pygame.draw.circle(self.screen, color, center, circ_radius)
      self.grid[next_pos] = 1
      self.idx_a = next_pos[0]
      self.idx_b = next_pos[1]
    return self.get_state()

  def simulate_step(self, action):
    color = colors[color_default]
    pos = (self.idx_a, self.idx_b)
    next_pos = GridWorld.move(pos, action)
    if self.grid[next_pos] == 0:
      center = GridWorld.compute_center(*pos)
      rect = pygame.rect.Rect((center[0] - a//2, center[1] - b//2), (a, b))
      next_screen = self.screen.copy()
      pygame.draw.rect(next_screen, bg_color, rect)
      center = GridWorld.compute_center(*next_pos)
      pygame.draw.circle(next_screen, color, center, circ_radius)
      frame = pygame.surfarray.pixels3d(next_screen)
      return GridWorld.to_float_tensor(frame), next_pos
    else:
      return self.get_state()

  @staticmethod
  def move(pos, action):
    tr = GridWorld.action2move(action)
    next_a = max(min(pos[0] + tr[0], n-1), 0)
    next_b = max(min(pos[1] + tr[1], n-1), 0)
    next_pos = (next_a, next_b)
    return next_pos

  @staticmethod
  def action2move(action):
    return moves[action]

  @staticmethod
  def to_float_tensor(frame):
    frame = torch.from_numpy(np.asarray(frame, dtype=np.float32) / 255.0).to(device)
    frame = torch.moveaxis(frame, 2, 0)
    return frame[None, ...]

  @staticmethod
  def compute_center(idx_a, idx_b):
    return (idx_a * a + a//2, idx_b * b + b//2)

  @staticmethod
  def abs_coords_pt(c: tuple, pt: tuple):
    return (c[0] + pt[0], c[1] + pt[1])

  @staticmethod
  def abs_coords_pts(c: tuple, pts: list):
    return [GridWorld.abs_coords_pt(c, pt) for pt in pts]


class ConvEncoder(nn.Module):

  def __init__(self):
    super(ConvEncoder, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=8, stride=4, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=5, stride=4, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=4, stride=3, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=False),
      nn.LeakyReLU(inplace=True),
    )

    for l in self.net.modules():
      if isinstance(l, nn.Conv2d):
        nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain("leaky_relu"))

  def forward(self, x):
    return self.net(x)


class ConvDecoder(nn.Module):

  def __init__(self):

    super(ConvDecoder, self).__init__()
    self.net = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(inplace=True),
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(inplace=True),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=3, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(inplace=True),
      nn.ConvTranspose2d(64, 32, kernel_size=5, stride=4, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(inplace=True),
      nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4, bias=True),
      nn.Sigmoid(),
    )
    for l in self.net.modules():
      if isinstance(l, nn.Conv2d):
        nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain("leaky_relu"))

  def forward(self, x):
    return self.net(x)


class VanillaAutoEncoder(nn.Module):

  def __init__(self):
    super(VanillaAutoEncoder, self).__init__()
    self.cv_enc = ConvEncoder()
    self.fc_enc = nn.Sequential(
      nn.Linear(512, 32),
      nn.LeakyReLU(inplace=True),
      nn.Linear(32, 2),
    )
    self.fc_dec = nn.Sequential(
      nn.Linear(2, 32),
      nn.LeakyReLU(inplace=True),
      nn.Linear(32, 512),
      nn.LeakyReLU(inplace=True),
    )
    self.cv_dec = ConvDecoder()

    for l in self.fc_enc.modules():
      if isinstance(l, nn.Linear):
        nn.init.xavier_normal_(l.weight)
        nn.init.zeros_(l.bias)

    for l in self.fc_dec.modules():
      if isinstance(l, nn.Linear):
        nn.init.xavier_normal_(l.weight)
        nn.init.zeros_(l.bias)

  def forward(self, x):
    x = self.cv_enc(x)
    x = torch.flatten(x, start_dim=1)
    h = self.fc_enc(x)
    x = self.fc_dec(h)
    x = x[..., None, None]
    x = self.cv_dec(x)
    return x, h

  def encoder(self, x):
    x = self.cv_enc(x)
    x = torch.flatten(x, start_dim=1)
    x = self.fc_enc(x)
    return x

  def test(self, x):
    x = self.cv_enc(x)
    x = torch.flatten(x, start_dim=1)
    h = self.fc_enc(x)
    x = self.fc_dec(h)
    x = x[..., None, None]
    x = self.cv_dec(x)
    return x, h


class FeaturePolicy(nn.Module):

  def __init__(self):
    super(FeaturePolicy, self).__init__()
    self.fc = nn.Linear(2,  4)

  def forward(self, x):
    x = self.fc(x)
    x = torch.softmax(x, dim=1)
    return x


class Buffer():

  def __init__(self):
    self.states = torch.empty((buffer_size, w, h, 3))
    self.poses = torch.empty((buffer_size, 2))
    self.latents = torch.empty((buffer_size, 2))
    self.actions = torch.empty((buffer_size,))
    self.s_idx = 0
    self.a_idx = 0
    self.p_idx = 0
    self.l_idx = 0

  def append_state(self, s):
    self.states[self.s_idx] = s
    self.s_idx = (self.s_idx + 1) % buffer_size

  def append_latent(self, l):
    self.latents[self.l_idx] = l
    self.l_idx = (self.a_idx + 1) % buffer_size

  def append_pos(self, p):
    self.actions[self.p_idx] = p
    self.p_idx = (self.a_idx + 1) % buffer_size

  def append_action(self, a):
    self.actions[self.a_idx] = a
    self.a_idx = (self.a_idx + 1) % buffer_size

  def clear(self):
    self.s_idx = 0
    self.a_idx = 0
    self.p_idx = 0
    self.l_idx = 0

vanilla_autoencoder = VanillaAutoEncoder().to(device)
ae_optimizer = torch.optim.Adam(params=vanilla_autoencoder.parameters(), lr=1e-4, weight_decay=0)
policies = [FeaturePolicy().to(device), FeaturePolicy().to(device)]
pc_optimizers = [torch.optim.Adam(params=policy.parameters(), lr=1e-4) for policy in policies]

def init_grid_world():
  grid_world = GridWorld()
  grid_world.initialize()
  return grid_world


def train():
  grid_world = init_grid_world()
  state, pos = grid_world.get_state()
  steps = 100000
  for i in range(steps):
    ae_optimizer.zero_grad()
    rstate, latent = vanilla_autoencoder.forward(state)
    rloss = torch.sum(0.5 * torch.square(rstate - state))
    rloss.backward()
    ae_optimizer.step()
    sloss = 0
    for k in range(2):
      ae_optimizer.zero_grad()
      pc_optimizers[k].zero_grad()
      rstate, latent = vanilla_autoencoder.forward(state)
      prob_actions = policies[k].forward(latent)[0]
      action = np.random.choice(action_space, p=prob_actions.detach().cpu().numpy())
      lprob_action = torch.log(prob_actions[action])
      next_state, next_pos = grid_world.simulate_step(action)
      next_rstate, next_latent = vanilla_autoencoder.forward(next_state)
      latent = torch.squeeze(latent, 0)
      next_latent = torch.squeeze(next_latent, 0)
      if not torch.equal(next_latent, latent):
        reward = torch.abs(next_latent[k] - latent[k]) / torch.sum(torch.abs(next_latent - latent))
        print(reward)
        grid_world.update_b(float(reward))
        sloss = -lprob_action * (reward - grid_world.get_b())
        sloss.backward()
        pc_optimizers[k].step()
        ae_optimizer.step()
    state, pos = grid_world.step(np.random.randint(4))
    if i % 100 == 0 and i > 0:
      print(pos, latent.detach().cpu().numpy().tolist())
      print(sloss)
      print(rloss)

if __name__ == '__main__':
  train()
