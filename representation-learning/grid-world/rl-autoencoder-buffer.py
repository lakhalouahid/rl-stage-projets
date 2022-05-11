from math import ceil
import os
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
n_features = 2
batch_size = 512

w, h, n = 80, 80, 5
a, b = w//n, h//n
aa, bb = a-2, b-2
eps = 0.25

_b = torch.zeros((n_features, n, n), dtype=torch.float32).to(device)
_n_b = torch.ones((n_features, n, n), dtype=torch.float32).to(device)

def get_b(poses, k):
  bs = torch.empty(size=(len(poses), 1), dtype=torch.float32).to(device)
  for i, pos in enumerate(poses):
    bs[i] = _b[k, pos[0], pos[1]]
  return bs

def update_b(poses, rewards, k):
  for i, pos in enumerate(poses):
    _b[k, pos[0], pos[1]] -= (_b[k, pos[0], pos[1]] - rewards[i][0]) / _n_b[k, pos[0], pos[1]]
    _n_b[k, pos[0], pos[1]] += 1.0

bg_color = (64, 64, 64)
action_space = (0, 1, 2, 3)
buffer_size = 1000


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
    self.frames = torch.empty((n, n, 3, w, h), dtype=torch.float32).to(device)

  def initialize(self):
    for _ in range(n_default):
      color = colors[np.random.randint(1, len(colors))]
      idxs_a, idxs_b = np.where(self.grid == 0)
      idxs = np.random.randint(len(idxs_a))
      idx_a, idx_b = idxs_a[idxs], idxs_b[idxs]
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

  def get_state_batch(self, batch_size):
    idxs_a, idxs_b = np.where(self.grid == 0)
    rand_idxs = torch.randperm(batch_size) % len(idxs_a)
    poses = list(zip(idxs_b[rand_idxs], idxs_b[rand_idxs]))
    frames = torch.empty(size=(batch_size, 3, w, h), dtype=torch.float32).to(device)
    for i, pos in enumerate(poses):
      frames[i] = self.frames[pos]
    return frames, poses



  def get_torch_frame(self):
    frame = pygame.surfarray.pixels3d(self.screen)
    return GridWorld.to_float_tensor(frame)

  def step(self, poses, actions, states):
    next_states = torch.empty(len(actions), 3, w, h).to(device)
    next_poses = []
    for i, (pos, action) in enumerate(zip(poses, actions)):
      next_pos = GridWorld.move(pos, action)
      if self.grid[next_pos] == 0:
        next_states[i] = self.frames[pos]
        next_poses.append(pos)
        # assert torch.equal(next_states[i], states[i]), "action didn't change the state, but next_state != state"
      else:
        next_states[i] = self.frames[next_pos]
        next_poses.append(next_pos)

    return next_states, next_poses

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

class ConvEncoder(nn.Module):

  def __init__(self):
    super(ConvEncoder, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=4, stride=4, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=4, stride=4, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(128, 256, kernel_size=2, stride=1, bias=False),
      nn.BatchNorm2d(256),
    )

    for l in self.net.modules():
      if isinstance(l, nn.Conv2d):
        nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain("leaky_relu"))

  def forward(self, x):
    return self.net(x)

  def beval(self):
    for l in self.net.modules():
      if isinstance(l, nn.BatchNorm2d):
        l.eval()

  def btrain(self):
    for l in self.net.modules():
      if isinstance(l, nn.BatchNorm2d):
        l.train()

class ConvDecoder(nn.Module):

  def __init__(self):

    super(ConvDecoder, self).__init__()
    self.net = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(32, 3, kernel_size=4, stride=4, bias=True),
      nn.Sigmoid(),
    )
    for l in self.net.modules():
      if isinstance(l, nn.Conv2d):
        nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain("leaky_relu"))

  def forward(self, x):
    return self.net(x)

  def beval(self):
    for l in self.net.modules():
      if isinstance(l, nn.BatchNorm2d):
        l.eval()

  def btrain(self):
    for l in self.net.modules():
      if isinstance(l, nn.BatchNorm2d):
        l.train()

class VanillaAutoEncoder(nn.Module):

  def __init__(self):
    super(VanillaAutoEncoder, self).__init__()
    self.cv_enc = ConvEncoder()
    self.fc_enc = nn.Sequential(
      nn.Linear(256, 32),
      nn.LeakyReLU(inplace=True),
      nn.Linear(32, 2),
    )
    self.fc_dec = nn.Sequential(
      nn.Linear(2, 32),
      nn.ReLU(inplace=True),
      nn.Linear(32, 256),
      nn.ReLU(inplace=True),
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

  def beval(self):
    self.cv_enc.beval()

  def btrain(self):
    self.cv_enc.btrain()

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
    x = (x + 0.001) / (1 + 0.001*4)
    return x

class MatchNetwork(nn.Module):
  def __init__(self):
    super(MatchNetwork, self).__init__()
    self.fc = nn.Linear(2, 2)

  def forward(self, x):
    return self.fc(x)

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
ae_optimizer = torch.optim.Adam(params=vanilla_autoencoder.parameters(), lr=1e-4)

match_network = MatchNetwork().to(device)
mn_optimizer = torch.optim.Adam(params=match_network.parameters(), lr=1e-4)

policies = [FeaturePolicy().to(device) for _ in range(n_features)]
pc_optimizers = [torch.optim.Adam(params=policy.parameters(), lr=1e-4) for policy in policies]

def init_grid_world():
  grid_world = GridWorld()
  grid_world.initialize()
  return grid_world

avg_w = 1000
batch_loop = 100000


def sample_actions(prob_actions):
  actions = torch.empty((batch_size, 1), dtype=torch.int64).to(device)
  for i in range(batch_size):
    actions[i][0] = np.random.choice(action_space, p=prob_actions[i].detach().cpu().numpy())
  return actions

kkk = 20
def train():
  grid_world = init_grid_world()
  for i in range(batch_loop):
    vanilla_autoencoder.btrain()
    ae_optimizer.zero_grad()
    states, poses = grid_world.get_state_batch(batch_size)
    rstates, _latents = vanilla_autoencoder.forward(states)
    rloss = torch.sum(torch.sum(0.5 * torch.square(rstates - states), dim=1))
    rloss.backward()
    ae_optimizer.step()
    for k in range(2):
      vanilla_autoencoder.beval()
      ae_optimizer.zero_grad()
      policies[k].zero_grad()
      latents = vanilla_autoencoder.encode(states.to(device))
      prob_actions = policies[k].forward(latents)
      actions = sample_actions(prob_actions)
      lprob_actions = torch.log(prob_actions.take_along_dim(actions, dim=1))
      next_states, _ = grid_world.step(poses, actions, states)
      next_latents = vanilla_autoencoder.encode(next_states.to(device))
      dlatents = torch.abs(next_latents - latents.detach())
      is_moved = torch.any(dlatents, dim=1)
      if torch.any(is_moved):
        c_poses = [poses[i] for i in range(len(poses)) if is_moved[i] == True]
        c_dlatents = dlatents[is_moved]
        c_lprob_actions = lprob_actions[is_moved]
        rewards  = c_dlatents[:, k:k+1] / torch.sum(c_dlatents, dim=1, keepdim=True)
        update_b(c_poses, rewards.detach(), k)
        mean_rewards = get_b(c_poses, k)
        sloss = torch.sum(-c_lprob_actions * (rewards - mean_rewards))
        sloss.backward()
        pc_optimizers[k].step()
        ae_optimizer.step()
        if i % kkk == 0:
          print(f"policky {k} --------")
          print(f"sloss: {sloss:.6f}")
          print(f"rloss: {rloss:.6f}")
    match_network.zero_grad()
    mn_error = match_network(_latents.detach()) - torch.tensor(poses, dtype=torch.float32).to(device)
    mn_loss = torch.sum(torch.sum(0.5 * torch.square(mn_error), dim=1))
    mn_loss.backward()
    mn_optimizer.step()
    logging.info(f"{rloss:.3f},{mn_loss:.3f}")
    if i % kkk == 0:
      print(f"mn_loss: {mn_loss:.3f}, len: {_latents.shape[0]}")
      print("---------------------------------------------------------")

if __name__ == '__main__':
  train()
