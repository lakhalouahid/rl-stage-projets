import numpy as np
import logging
import argparse

from time import time
from blackjack import policy_improve, policy_eval, policy_test
from matplotlib import cm, pyplot as plt




logging.basicConfig(filename=f"logs/evaluation-{int(time())}", format="%(message)s", level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--accuracy", type=float, default=1e-4, help="Accuracy of the evaluation")



player_sums = np.arange(12, 22)
dealer_displayed_cards = np.arange(1, 11)
titles = [
  "Value function with no usable ace and semi pure random card sampling",
  "Value function with no usable ace and pure random card sampling",
  "Value function with usable ace and semi pure random card sampling",
  "Value function with usable ace and pure random card sampling"
  ]


def label_axe(fig, ax, V, title):
  img = ax.imshow(V)
  ax.set_yticks(np.arange(player_sums.shape[0]), labels=player_sums)
  ax.set_xticks(np.arange(dealer_displayed_cards.shape[0]), labels=dealer_displayed_cards)
  ax.set_title(title, fontsize='xx-large')
  ax.set_xlabel("Dealer showed card 1 .. 10", fontsize='xx-large')
  ax.set_ylabel("Player total sum card 12 .. 21", fontsize='xx-large')
  cbar = ax.figure.colorbar(img, ax=ax)
  cbar.ax.set_ylabel("State value", rotation=-90, va="bottom", fontsize='xx-large')
  for i in range(player_sums.shape[0]):
      for j in range(dealer_displayed_cards.shape[0]):
        ax.text(j, i, f"{V[i, j]:.3f}", ha="center", va="center", color="w", fontsize='x-large')


def visualize_value( A: np.ndarray, data: list[np.ndarray]) -> None:
  V00, V01, V10, V11  = data

  fig1  = plt.figure(1, frameon=False)
  ax1 = fig1.add_subplot(1, 1, 1)
  label_axe(fig1, ax1, V00, titles[0])
  fig1.tight_layout()

  fig2  = plt.figure(2, frameon=False)
  ax2 = fig2.add_subplot(1, 1, 1)
  label_axe(fig2, ax2, V01, titles[1])
  fig2.tight_layout()

  fig3  = plt.figure(3, frameon=False)
  ax3 = fig3.add_subplot(1, 1, 1)
  label_axe(fig3, ax3, V10, titles[2])
  fig3.tight_layout()

  fig4  = plt.figure(4, frameon=False)
  ax4 = fig4.add_subplot(1, 1, 1)
  label_axe(fig4, ax4, V11, titles[3])
  fig4.tight_layout()

  plt.show()
  fig1.savefig("./images/V00.png")
  fig2.savefig("./images/V01.png")
  fig3.savefig("./images/V10.png")
  fig4.savefig("./images/V11.png")




def main():
  ## policy evaluation
  acc = parser.parse_args().accuracy
  A = np.zeros((10, 10), dtype=np.int8)
  A[:8, :] = 1
  V11, nV11 = policy_eval(A, usable_ace=True, accuracy=acc, pure_rand=True)
  V01, nV01 = policy_eval(A, usable_ace=False, accuracy=acc, pure_rand=True)
  V10, nV10 = policy_eval(A, usable_ace=True, accuracy=acc, pure_rand=False)
  V00, nV00 = policy_eval(A, usable_ace=False, accuracy=acc, pure_rand=False)
  logging.info(f"{V00}\n{nV00}")
  logging.info(f"{V01}\n{nV01}")
  logging.info(f"{V10}\n{nV10}")
  logging.info(f"{V11}\n{nV11}")
  visualize_value(A, [V00, V01, V10, V11])


if __name__ == '__main__':
  main()


