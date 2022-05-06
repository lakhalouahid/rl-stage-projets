import os
import torch
import logging
import numpy as np



from time import sleep, time
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda:0")
logging.basicConfig(filename=f"logs/plain-autoencoder-{time():.0f}.log", format="%(message)s", level=logging.INFO)

class AE(nn.Module):
  def __init__(self):
    super(AE, self).__init__()
    self.cv_enc = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=8, stride=4, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=5, stride=4, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=4, stride=3, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
    )
    self.fc_enc = nn.Sequential(
      nn.Linear(512, 32),
      nn.ReLU(inplace=True),
      nn.Linear(32, 2),
    )
    self.fc_dec = nn.Sequential(
      nn.Linear(2, 32),
      nn.ReLU(inplace=True),
      nn.Linear(32, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
    )
    self.cv_dec = nn.Sequential(
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
    )

  def forward(self, x):
    x = self.cv_enc(x)
    x = torch.flatten(x, start_dim=1)
    x = self.fc_enc(x)
    x = self.fc_dec(x)
    x = x[..., None, None]
    x = self.cv_dec(x)
    return x

  def test(self, x):
    x = self.cv_enc(x)
    x = torch.flatten(x, start_dim=1)
    code = self.fc_enc(x)
    x = self.fc_dec(code)
    x = x[..., None, None]
    x = self.cv_dec(x)
    return x, code



ae = AE().to(device)
opt = torch.optim.Adam(params=ae.parameters(), lr=1e-3, weight_decay=0)

dataset = []
dataset_folder = "data/dataset"

for image_name in os.listdir(dataset_folder):
  image_path = os.path.join(dataset_folder, image_name)
  image = Image.open(image_path)
  image = transforms.functional.to_tensor(image)
  dataset.append(image)

images = torch.stack(dataset).to(device)

def train():
  ae.train()
  for epoch_idx in range(1<<16):
    opt.zero_grad()
    loss = 0.5 * torch.sum(torch.square(images - ae(images)))
    if epoch_idx & 7 == 0:
      logging.info(f"epoch {epoch_idx + 1} with loss: {loss:.6f}")
      print(f"epoch {epoch_idx + 1} with loss: {loss:.6f}")
    loss.backward()
    if epoch_idx & 1023 == 0 and epoch_idx > 0:
      filename = os.path.join("checkpoints", f"plain-ae-{time():.0f}-{float(loss):0.5f}.pt")
      torch.save(ae.state_dict(), filename)
    opt.step()

def test():
  ae.load_state_dict(torch.load("./checkpoints/ae-dict-0.00065.pt"))
  image = images[0]
  ae.eval()
  with torch.no_grad():
    rc_image, code = ae.test(image[None, ...])
  print(code)
  # transforms.functional.to_pil_image(image.cpu()).show()
  # transforms.functional.to_pil_image(rc_image[0].cpu()).show()
  # sleep(0.2)


if __name__ == '__main__':
  train()

