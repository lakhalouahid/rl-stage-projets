import numpy as np
import argparse
import matplotlib.pyplot as plt

n = 5

def visualize(frames):
  f, axes = plt.subplots(n, n)
  for i in range(n):
    for j in range(n):
      axes[i][j].imshow(self.frames[i, j].moveaxis(0, 2).cpu().numpy())
  plt.show(block=False)
