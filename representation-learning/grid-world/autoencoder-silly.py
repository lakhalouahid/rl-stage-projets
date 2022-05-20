import os
import pdb
import argparse
import logging
import torch
import pygame
import debugnn

import numpy as np
import matplotlib.pyplot as plt


from time import sleep, time
from torch import nn

batchsize=25
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--plain", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--lbd", type=float, default=1.0)
parser.add_argument("--maxlr", type=float, default=5e-3)
parser.add_argument("--minlr", type=float, default=5e-4)
parser.add_argument("--lrdecay", type=float, default=0.9999)
parser.add_argument("--weightdecay", type=float, default=0.9995)
parser.add_argument("--loop", type=int, default=250000)
args = parser.parse_args()
device = torch.device("cuda:0")
root = os.getcwd()
logfile = os.path.join(root, "logs/train.log")

if not os.path.exists(os.path.join(root, "logs")):
  os.mkdir(os.path.join(root, "logs"))

if not os.path.exists(os.path.join(root, "checkpoints")):
  os.mkdir(os.path.join(root, "checkpoints"))

if not os.path.exists(os.path.join(root, "data")):
  os.mkdir(os.path.join(root, "data"))
if not args.test:
  logging.basicConfig(filename=logfile, format="%(message)s", level=logging.INFO)

w, h, n = 5, 5, 5
a, b = w//n, h//n

action_space = (0, 1, 2, 3)

class GW():
  def __init__(self):
    self.frames = torch.zeros((n, n, 1, w, h), dtype=torch.float32).to(device)

  def initialize(self):
    for idx_a in range(n):
      for idx_b in range(n):
        self.frames[idx_a, idx_b, 0, idx_a, idx_b] = 1.0

  def visualize(self):
    f, axes = plt.subplots(n, n)
    for i in range(n):
      for j in range(n):
        axes[i][j].imshow(self.frames[i, j].moveaxis(0, 2).cpu().numpy())
    plt.show(block=False)

  @staticmethod
  def visualize_frames(frames, size):
    f, axes = plt.subplots(size[0], size[1])
    k = 0
    for i, j in zip(*[shit.flat for shit in np.mgrid[0:size[0], 0:size[1]]]):
      axes[i][j].imshow(frames[k].moveaxis(0, 2).cpu().numpy())
      k += 1
      if k == len(frames):
        break
    plt.show(block=False)

  def sample(self):
    sampled_flat_indexes = np.random.permutation(n*n)
    poses = np.unravel_index(sampled_flat_indexes, shape=(n, n))
    frames = self.frames[poses]
    return frames, poses

  def step(self, poses, actions):
    actions = actions.detach().cpu().numpy()
    actions = (actions[:, 0], actions[:, 1])
    next_poses = tuple([(_poses + _actions + n) % n for _poses, _actions in zip(poses, actions)])
    frames = self.frames[next_poses]
    return frames, poses



class ConvEncoder(nn.Module):
  def __init__(self):
    super(ConvEncoder, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(1, 4, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(4),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(4, 8, kernel_size=2, stride=2, bias=False),
      nn.BatchNorm2d(8),
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
      nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, bias=False),
      nn.BatchNorm2d(4),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, bias=True),
      nn.Sigmoid(),
    )
    for l in self.net.modules():
      if isinstance(l, nn.Conv2d):
        nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain("relu"))

  def forward(self, x):
    return self.net(x)


class VanillaAutoEncoder(nn.Module):

  def __init__(self):
    super(VanillaAutoEncoder, self).__init__()
    self.cv_enc = ConvEncoder()
    self.fc_enc = nn.Sequential(
      nn.Linear(8, 2),
    )
    self.fc_dec = nn.Sequential(
      nn.Linear(2, 8),
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

  def encode(self, x):
    x = self.cv_enc(x)
    x = torch.flatten(x, start_dim=1)
    x = self.fc_enc(x)
    return x


class FeaturePolicy(nn.Module):

  def __init__(self):
    super(FeaturePolicy, self).__init__()
    self.fc = nn.Sequential(
        nn.Linear(2, 4),
    )

  def forward(self, x):
    x = self.fc(x)
    x = torch.softmax(x, dim=1)
    x = (x + 0.01) / (1 + 0.01*4)
    return x
vanilla_autoencoder = VanillaAutoEncoder().to(device)
ae_optimizer = torch.optim.Adam(params=vanilla_autoencoder.parameters(), lr=args.maxlr, weight_decay=args.weightdecay)
ae_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=ae_optimizer, gamma=args.lrdecay)

policies = [FeaturePolicy().to(device) for _ in range(2)]
pc_optimizers = [torch.optim.Adam(params=policy.parameters(), lr=args.maxlr, weight_decay=args.weightdecay) for policy in policies]
pc_schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer=pc_optimizer, gamma=args.lrdecay) for pc_optimizer in pc_optimizers]

def init_grid_world():
  grid_world = GW()
  grid_world.initialize()
  return grid_world


def sample_actions(prob_actions):
  c = prob_actions.detach().cpu().numpy().cumsum(axis=1)
  u = np.random.rand(len(c), 1)
  choices = (u < c).argmax(axis=1)
  return torch.from_numpy(choices).to(device)

moves = torch.tensor([
  [-1, 0],
  [0, -1],
  [+1, 0],
  [0, +1],
])

def actions2move(actions, rlossistoobig):
  if rlossistoobig:
    return moves[actions]
  else:
    directions = moves[actions]
    return directions * torch.randint_like(directions, 1, n)

grid_world = init_grid_world()

def train():
  for i in range(args.loop):
    vanilla_autoencoder.train()
    states, poses = grid_world.sample()
    rstates, _latents = vanilla_autoencoder.forward(states)
    rloss = torch.sum(torch.sum(0.5 * torch.square(rstates - states), dim=1))
    if bool(torch.isnan(rloss)):
      break
    rloss.backward()
    if args.plain:
      ae_optimizer.step()
      ae_optimizer.zero_grad()
    else:
      sloss = [0.0, 0.0]
      mean_rewards = [0.0, 0.0]
      vanilla_autoencoder.eval()
      for k in range(2):
        latents = vanilla_autoencoder.encode(states.to(device))
        prob_actions = policies[k].forward(latents)
        actions = sample_actions(prob_actions)
        lprob_actions = torch.log(prob_actions.take_along_dim(actions[..., None], dim=1))
        _moves = actions2move(actions, float(rloss) > 1600)
        next_states, _ = grid_world.step(poses, _moves)
        next_latents = vanilla_autoencoder.encode(next_states.to(device))
        dlatents = torch.abs(next_latents - latents.detach())
        rewards  = dlatents[:, k:k+1] / torch.sum(dlatents, dim=1, keepdim=True)
        mean_rewards[k] = torch.mean(rewards)
        sloss[k] = torch.sum(-lprob_actions * rewards ) * args.lbd
        sloss[k].backward()
        if i % 100 == 0:
          logging.info(f"{rloss},{sloss[0]},{sloss[0]}")
          print(f"policy loss {k}: {sloss[k]:.3f}")
      pc_optimizers[0].step()
      pc_optimizers[1].step()
      ae_optimizer.step()
      pc_optimizers[0].zero_grad()
      pc_optimizers[1].zero_grad()
      ae_optimizer.zero_grad()
      if ae_scheduler.get_lr()[0] > args.minlr:
        ae_scheduler.step()
        pc_optimizers[0].step()
        pc_optimizers[1].step()
    if i % 100 == 0:
      training_traceback = {
          "loop": i,
          "lr": {
            "ae_optimizer": ae_optimizer.param_groups[0]['lr'],
            "pc_optimizer": [
                pc_optimizers[0].param_groups[0]['lr'],
                pc_optimizers[1].param_groups[0]['lr']
              ]
            }
          }
      debugnn.json_write(training_traceback, "status.json")
      print(f"r_loss: {rloss:.3f}")
      print("---------------------------------------------------------")

    if i % 10000 == 0 :
      filename = os.path.join("checkpoints", f"{rloss}.pt")
      torch.save(vanilla_autoencoder.state_dict(), filename)

def resume():
  global vanilla_autoencoder, args, ae_optimizer, pc_optimizers
  statedictfile = debugnn.get_latesfile("checkpoints")
  vanilla_autoencoder.load_state_dict(torch.load(statedictfile))
  tcb = debugnn.json_read("status.json")
  args.loop = tcb["loop"]
  for g in ae_optimizer.param_groups:
    g['lr'] = tcb["lr"]["ae_optimizer"]
  for g in pc_optimizers[0].param_groups:
    g['lr'] = tcb["lr"]["pc_optimizer"][0]
  for g in pc_optimizers[1].param_groups:
    g['lr'] = tcb["lr"]["pc_optimizer"][1]

def visualise_latents(latents):
  f, axe = plt.subplots()
  X = latents[:, 0]
  Y = latents[:, 1]
  axe.scatter(X, Y, s=100, color="red")
  axe.set_xlabel("X")
  axe.set_ylabel("Y")
  for i in range(batchsize):
    idx = np.unravel_index(i, (n, n))
    axe.text(X[i], Y[i], f"({idx[0]}, {idx[1]})")
  plt.show(block=False)

def visualise_latents2(latents):
  f, axe = plt.subplots()
  X = latents[:, 0]
  Y = latents[:, 1]
  Xi, Yi = np.unravel_index(np.arange(n*n), (n, n))
  axe.scatter(Xi, Yi, s=100, color="red")
  axe.set_xlabel("X")
  axe.set_ylabel("Y")
  for i in range(batchsize):
    idx = np.unravel_index(i, (n, n))
    axe.text(idx[0], idx[1], f"({X[i]:.2f}, {Y[i]:.2f})")
  plt.show(block=False)


def test():
  fzf = pyfzf.FzfPrompt("/usr/bin/fzf")
  while True:
    uinput = input("Enter the commands (s/q): ")
    if uinput == "q":
      break
    state_dict_file = fzf.prompt(os.listdir(os.path.join("checkpoints")))[0]
    vanilla_autoencoder.load_state_dict(torch.load("checkpoints/" + state_dict_file))
    frames = grid_world.frames.flatten(start_dim=0, end_dim=1)
    rframes, latents = vanilla_autoencoder(frames)
    grid_world.visualize_frames(frames.detach().cpu(), (n, n))
    grid_world.visualize_frames(rframes.detach().cpu(), (n, n))
    np_latents = latents.detach().cpu().numpy()
    visualise_latents(np_latents)
    visualise_latents2(np_latents)

if __name__ == '__main__':
  if args.test:
    test()
  if args.resume:
    resume()
    train()
  else:
    train()
