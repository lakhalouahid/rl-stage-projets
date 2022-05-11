import numpy as np
import os
import matplotlib.pyplot as plt


def get_ctime(path):
  return os.stat(path).st_ctime

logfiles = ["logs/"+ file for file in os.listdir("logs")]
logfiles.sort(key=get_ctime, reverse=True)
csv_filename = logfiles[0]

data = np.genfromtxt(csv_filename, delimiter=",")

plt.subplot(211)
plt.plot(data[:, 0])
plt.subplot(212)
plt.plot(data[:, 1])

plt.show()
