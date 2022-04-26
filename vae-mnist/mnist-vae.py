import logging
import torch
import torchvision


import numpy as np

from time import time
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import transforms



lr = 1e-3
z_dim = 4
x_dim = 784
device = torch.device("cuda")

transform = transforms.Compose([
  transforms.ToTensor(),
])

class Encoder(nn.Module):
  def __init__(self) -> None:
    super(Encoder, self).__init__()
    self.ln1 = nn.Linear(x_dim, 256)
    self.ln2 = nn.Linear(256, 128)
    self.ln3 = nn.Linear(128, 32)
    self.ln4 = nn.Linear(32, 8)

  def forward(self, x):
    x = F.relu(self.ln1(x), inplace=True)
    x = F.relu(self.ln2(x), inplace=True)
    x = F.relu(self.ln3(x), inplace=True)
    x = self.ln4(x)
    return x


class Decoder(nn.Module):
  def __init__(self) -> None:
    super(Decoder, self).__init__()
    self.ln1 = nn.Linear(4, 32)
    self.ln2 = nn.Linear(32, 128)
    self.ln3 = nn.Linear(128, 256)
    self.ln4 = nn.Linear(256, 784)

  def forward(self, x):
    x = F.relu(self.ln1(x), inplace=True)
    x = F.relu(self.ln2(x), inplace=True)
    x = F.relu(self.ln3(x), inplace=True)
    x = torch.sigmoid(self.ln4(x))
    return x



class AutoEncoder(nn.Module):
  def __init__(self) -> None:
    super(AutoEncoder, self).__init__()
    self.enc = Encoder()
    self.dec = Decoder()

  def sample(self, x):
    xstd = torch.as_tensor(np.random.multivariate_normal(mean=np.zeros(shape=(4,)), cov=np.eye(4), size=(x.shape[0],)), dtype=torch.float32).to(device)
    return x[:, :4] + torch.exp(x[:, 4:] / 2) * xstd

  def forward(self, x):
    x = self.enc(x)
    x = self.sample(x)
    x = self.dec(x)
    return x

  def encode(self, x):
    return self.enc(x)

  def decode(self, x):
    return self.dec(x)

def train(net, optim, sched, dataloader, n_epochs=4):
  total_loss_old = 1e9
  for epoch_idx in range(n_epochs):
    total_loss = 0
    n = 1
    print(f'Epoch {epoch_idx + 1} -------------------------------------------------------------')
    for b_ix, b in enumerate(dataloader):
      X, _  = b
      X = torch.flatten(X, start_dim=1).to(device)
      optim.zero_grad()
      z = net.encode(X)
      Xout = net.decode(net.sample(z))
      likeli_loss = 0.5 * torch.mean(torch.sum(torch.square(Xout - X), dim=1))
      kldiv_loss = 0.5 * torch.mean(
          torch.sum(torch.exp(z[:, 4:]) + \
          torch.square(z[:, 4:]) - \
          torch.ones_like(z[:, :4]) - \
          z[:, 4:], dim=1))
      loss = likeli_loss + kldiv_loss
      total_loss += (float(loss) - total_loss) / n
      n += 1

      if b_ix & 255 == 0:
        print(float(loss))

      loss.backward()
      optim.step()

    if total_loss < total_loss_old:
      total_loss_old = total_loss
      torch.save(net.state_dict(), f"checkpoints/net-state-dict-{total_loss_old:2.2f}.pt")
    sched.step()


def main():
  train_dataset = torchvision.datasets.MNIST('datasets', train=True, download=True, transform=transform)
  dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

  net = AutoEncoder().to(device)
  optim = torch.optim.Adam(params=net.parameters(), lr=lr, )
  sched =torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)
  train(net, optim,  sched, dataloader, n_epochs=140)

def test():
  test_dataset = torchvision.datasets.MNIST('datasets', train=False, download=True, transform=transform)
  dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

  net = AutoEncoder().to(device)

  net.load_state_dict(torch.load("./checkpoints/net-state-dict-8.36.pt"))
  while True:
    with torch.no_grad():
      z = torch.as_tensor(np.random.multivariate_normal(mean=np.zeros(shape=(4,)), cov=np.eye(4), size=(1,)), dtype=torch.float32).to(device)
      Xout = net.decode(z)
      Xout_ = Xout.reshape(shape=(28, 28, 1)).cpu().detach().numpy()
      plt.imshow(Xout_)
      plt.show()

if __name__ == '__main__':
 test()
