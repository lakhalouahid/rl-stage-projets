from typing import Optional
from os import getcwd
from os.path import join
import numpy as np
import matplotlib.pyplot as plt



def make_avg(vector: np.ndarray, window: Optional[int]=None):
  avg_vector = np.empty_like(vector)
  if not window:
    window = min(len(vector), 100)
  ravg = 0
  n = 1.0
  for i in range(len(vector)):
    ravg += (vector[i] - ravg) / n
    n = max(n + 1, window)
    avg_vector[i] = ravg
  return avg_vector




def main():
  data = np.loadtxt(join(getcwd(), "logs/train.log"), delimiter=",", dtype=float)
  titles = ["Reconstruction Loss Graph", "Policy loss n° 1 Graph", "Policy loss n° 2 Graph", "Reward n° 1 Plot", "Reward n° 2 Plot"]
  for i in range(data.shape[1]):
    f, ax = plt.subplots()
    if i in [1, 2, 3, 4]:
      vector = make_avg(data[:, i], window=100)
    else:
      vector = data[:, i]
    ax.plot(vector)
    ax.set_title(titles[i])
  plt.show(block=False)
  input()

if __name__  == "__main__":
  main()
