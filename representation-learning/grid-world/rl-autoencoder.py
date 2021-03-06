from math import ceil
import os
import argparse
import pdb
import torch
import pygame
import logging
import numpy as np



from time import time
from torch import nn
from torchvision import transforms

device = torch.device("cuda:0")
logging.basicConfig(filename=f"logs/plain-autoencoder-{time():.0f}.log", format="%(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rand", action="store_true")
parser.add_argument("-l", "--lbd", type=float, default=1.0)
args = parser.parse_args()

w, h, n = 360, 360, 6
a, b = w//n, h//n
aa, bb = a-8, b-8
lbd = args.lbd
eps = 0.25


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
  (-b//2, 0),
  (0, -a//2),
  (+b//2, 0),
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

n_default = 10
color_default = 0
n_samples = 1000

class GridWorld():

  def __init__(self):
    self.grid = np.zeros((n, n), dtype=np.uint8)
    self.screen = pygame.Surface((w, h))
    self.screen.fill(bg_color)
    self.b = torch.zeros((n, n), dtype=torch.float32)
    self.n_b = torch.ones((n, n), dtype=torch.float32)
    self.frames = torch.empty((n, n, 3, w, h), dtype=torch.float32).to(device)

  def update_b(self, r):
    self.b[self.idx_a, self.idx_b] -= (self.b[self.idx_a, self.idx_b] - r) / self.n_b[self.idx_a, self.idx_b]
    self.n_b[self.idx_a, self.idx_b] += 1.0

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

    idxs_a, idxs_b = np.where(self.grid == 0)
    color = colors[color_default]
    for idx_a, idx_b in zip(idxs_a, idxs_b):
      center = GridWorld.compute_center(idx_a, idx_b)
      pygame.draw.circle(self.screen, color, center, circ_radius)
      self.frames[idx_a, idx_b] = self.get_torch_frame()
      rect = pygame.rect.Rect((center[0] - a//2, center[1] - b//2), (a, b))
      pygame.draw.rect(self.screen, bg_color, rect)

    idxs_a, idxs_b = np.where(self.grid == 0)
    idx = np.random.randint(len(idxs_a))
    self.idx_a, self.idx_b = idxs_a[idx], idxs_b[idx]
    self.grid[self.idx_a, self.idx_b] = 1
    return self.get_state()

  def get_state(self):
    return self.frames[self.idx_a, self.idx_b], (self.idx_a, self.idx_b)

  def get_torch_frame(self):
    frame = pygame.surfarray.pixels3d(self.screen)
    return GridWorld.to_float_tensor(frame)

  def step(self, action):
    pos = (self.idx_a, self.idx_b)
    next_pos = GridWorld.move(pos, action)
    if self.grid[next_pos] == 0:
      self.grid[pos] = 0
      self.grid[next_pos] = 1
      self.idx_a = next_pos[0]
      self.idx_b = next_pos[1]
    return self.get_state()

  def simulate_step(self, action):
    pos = (self.idx_a, self.idx_b)
    next_pos = GridWorld.move(pos, action)
    if self.grid[next_pos] == 0:
      return self.frames[next_pos[0], next_pos[1]], next_pos
    else:
      return self.get_state()

  @staticmethod
  def move(pos, action):
    tr = GridWorld.action2move(action)
    next_a = max(min(pos[0] + tr[0], n-1), 0)
    next_b = max(min(pos[1] + tr[1], n-1), 0)
    return (next_a, next_b)

  @staticmethod
  def action2move(action):
    return moves[action]

  @staticmethod
  def to_float_tensor(frame):
    frame = torch.from_numpy(np.asarray(frame, dtype=np.float32) / 255.0)
    frame = torch.moveaxis(frame, 2, 0)
    return frame

  @staticmethod
  def compute_center(idx_a, idx_b):
    return (idx_a * a + a//2, idx_b * b + b//2)

  @staticmethod
  def abs_coords_pt(c: tuple, pt: tuple):
    return (c[0] + pt[0], c[1] + pt[1])

  @staticmethod
  def abs_coords_pts(c: tuple, pts: list):
    return [GridWorld.abs_coords_pt(c, pt) for pt in pts]

class MatchNetwork(nn.Module):
  def __init__(self):
    super(MatchNetwork, self).__init__()
    self.fc = nn.Linear(2, 2)

  def forward(self, x):
    return self.fc(x)

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
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=3, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(64, 32, kernel_size=5, stride=4, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
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

  def encode(self, x):
    x = self.cv_enc(x)
    x = torch.flatten(x, start_dim=1)
    x = self.fc_enc(x)
    return x


class FeaturePolicy(nn.Module):

  def __init__(self):
    super(FeaturePolicy, self).__init__()
    self.fc = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 4),
    )

  def forward(self, x):
    x = self.fc(x)
    x = torch.softmax(x, dim=1)
    x = (x + 0.01) / (1 + 0.04)
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

n_features = 2

vanilla_autoencoder = VanillaAutoEncoder().to(device)
ae_optimizer = torch.optim.Adam(params=vanilla_autoencoder.parameters(), lr=3e-5)

match_network = MatchNetwork()
mn_optimizer = torch.optim.Adam(params=match_network.parameters(), lr=3e-4)

policies = [FeaturePolicy().to(device) for _ in range(n_features)]
pc_optimizers = [torch.optim.Adam(params=policy.parameters(), lr=3e-5) for policy in policies]

def init_grid_world():
  grid_world = GridWorld()
  grid_world.initialize()
  return grid_world

avg_w = 1000

def train():
  grid_world = init_grid_world()
  state, pos = grid_world.get_state()
  steps = 1000000
  sloss_avg = [0.0, 0.0]
  reward_avg = [0.0, 0.0]
  rloss_avg = 0.0
  mloss_avg = 0.0
  for i in range(steps):
    vanilla_autoencoder.zero_grad()
    rstate, latent = vanilla_autoencoder.forward(state[None, ...])
    rloss = torch.sum(0.5 * torch.square(rstate - state))
    rloss_avg -= (rloss_avg - float(rloss)) / avg_w
    rloss.backward()
    ae_optimizer.step()
    reward = 0
    for k in range(2):
      vanilla_autoencoder.zero_grad()
      policies[k].zero_grad()
      latent = vanilla_autoencoder.encode(state[None, ...])
      prob_actions = policies[k].forward(latent)[0]
      if np.random.random() < eps and args.rand == False:
        action = np.random.choice(action_space)
      else:
        action = np.random.choice(action_space, p=prob_actions.detach().cpu().numpy())
      lprob_action = torch.log(prob_actions[action])
      next_state, _ = grid_world.simulate_step(action)
      next_latent = vanilla_autoencoder.encode(next_state[None, ...])
      latent = torch.squeeze(latent, 0)
      next_latent = torch.squeeze(next_latent, 0)

      if not torch.equal(next_latent, latent):
        reward = torch.abs(next_latent[k] - latent[k]) / torch.sum(torch.abs(next_latent - latent))
        reward_avg[k] -= (reward_avg[k] - float(reward)) / avg_w
        grid_world.update_b(float(reward))
        sloss = -lprob_action * (reward - grid_world.get_b()) * lbd
        sloss_avg[k] -= (sloss_avg[k] - float(sloss)) / avg_w
        sloss.backward()
        pc_optimizers[k].step()
        ae_optimizer.step()
        if i % 25 == 0:
          print(f"policky {k} --------")
          print(f"reward: {reward_avg[k]:.2f}")
          print(f"sloss: {sloss_avg[k]:.6f}")
          print(f"rloss: {rloss_avg:.3f}")
    state, pos = grid_world.step(np.random.randint(4))
    match_network.zero_grad()
    mn_loss = torch.sum(0.5 * torch.square(match_network(latent.detach().cpu()) - torch.tensor(pos, dtype=torch.float32)))
    mn_loss.backward()
    mloss_avg -= (mloss_avg - float(mn_loss)) / 1000
    mn_optimizer.step()
    if i % 25 == 0 and reward != 0:
      print(f"mn_loss: {mloss_avg:.3f}")
      print("---------------------------------------------------------")

if __name__ == '__main__':
  train()
