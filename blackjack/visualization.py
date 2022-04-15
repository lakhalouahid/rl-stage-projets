import numpy as np
import logging
import argparse
from numba import jit, prange

from time import time
from blackjack import policy_improve, policy_eval, policy_test
from matplotlib import cm, pyplot as plt




parser = argparse.ArgumentParser()
parser.add_argument("-a", "--accuracy", type=float, default=1e-4, help="Accuracy of the evaluation")
parser.add_argument("-e", "--eval", action="store_true", help="Accuracy of the evaluation")



player_sums = np.arange(12, 22)
dealer_displayed_cards = np.arange(1, 11)
value_titles = [
  "Value function with no usable ace and semi pure random card sampling",
  "Value function with no usable ace and pure random card sampling",
  "Value function with usable ace and semi pure random card sampling",
  "Value function with usable ace and pure random card sampling"
]

policy_titles = [
  "Policy function with no usable ace and semi pure random card sampling",
  "Policy function with no usable ace and pure random card sampling",
  "Policy function with usable ace and semi pure random card sampling",
  "Policy function with usable ace and pure random card sampling"
]

def label_axe_eval(fig, ax, V, title):
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

def label_axe_improve(fig, ax, A, title):
  img = ax.imshow(A)
  ax.set_yticks(np.arange(player_sums.shape[0]), labels=player_sums)
  ax.set_xticks(np.arange(dealer_displayed_cards.shape[0]), labels=dealer_displayed_cards)
  ax.set_title(title, fontsize='xx-large')
  ax.set_xlabel("Dealer showed card 1 .. 10", fontsize='xx-large')
  ax.set_ylabel("Player total sum card 12 .. 21", fontsize='xx-large')
  for i in range(player_sums.shape[0]):
      for j in range(dealer_displayed_cards.shape[0]):
        ax.text(j, i, A[i, j], ha="center", va="center", color="w", fontsize='x-large')

def visualize_eval(A: np.ndarray, data: list[np.ndarray]) -> None:
  V00, V01, V10, V11  = data

  fig1  = plt.figure(1, frameon=False)
  ax1 = fig1.add_subplot(1, 1, 1)
  label_axe_eval(fig1, ax1, V00, value_titles[0])
  fig1.tight_layout()

  fig2  = plt.figure(2, frameon=False)
  ax2 = fig2.add_subplot(1, 1, 1)
  label_axe_eval(fig2, ax2, V01, value_titles[1])
  fig2.tight_layout()

  fig3  = plt.figure(3, frameon=False)
  ax3 = fig3.add_subplot(1, 1, 1)
  label_axe_eval(fig3, ax3, V10, value_titles[2])
  fig3.tight_layout()

  fig4  = plt.figure(4, frameon=False)
  ax4 = fig4.add_subplot(1, 1, 1)
  label_axe_eval(fig4, ax4, V11, value_titles[3])
  fig4.tight_layout()

  plt.show()
  fig1.savefig("./images/V00.png")
  fig2.savefig("./images/V01.png")
  fig3.savefig("./images/V10.png")
  fig4.savefig("./images/V11.png")




@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_eval(A: np.ndarray, acc: float) -> list[np.ndarray]:
  V = np.zeros((4, 10, 10), dtype=np.float32)
  usable_ace_list = [True, False, True, False]
  pure_rand_list = [True, False, True, False]
  for i in prange(4):
    V[i, :, :] = policy_eval(A, usable_ace=usable_ace_list[i], accuracy=acc, pure_rand=pure_rand_list[i])[0]
  return [V[0, :, :], V[1, :, :], V[2, :, :], V[3, :, :]]

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_improve(acc: float) -> list[np.ndarray]:
  A = np.zeros((4, 10, 10), dtype=np.float32)
  usable_ace_list = [True, False, True, False]
  pure_rand_list = [True, False, True, False]
  for i in prange(4):
    A[i, :, :] = policy_improve(usable_ace=usable_ace_list[i], accuracy=acc, pure_rand=pure_rand_list[i])[0]
  return [A[0, :, :], A[1, :, :], A[2, :, :], A[3, :, :]]

def evaluation(args):
  acc = args.accuracy
  logging.basicConfig(filename=f"logs/evaluation-{int(time())}", format="%(message)s", level=logging.INFO)
  A = np.zeros((10, 10), dtype=np.int8)
  A[:8, :] = 1
  V00, V01, V10, V11 = compute_eval(A, acc)
  logging.info(f"{V00}")
  logging.info(f"{V01}")
  logging.info(f"{V10}")
  logging.info(f"{V11}")
  visualize_eval(A, [V00, V01, V10, V11])

def improve(args):
  acc = args.accuracy
  logging.basicConfig(filename=f"logs/evaluation-{int(time())}", format="%(message)s", level=logging.INFO)
  A11 = policy_improve(usable_ace=True, accuracy=acc, pure_rand=True)
  A01 = policy_improve(usable_ace=False, accuracy=acc, pure_rand=True)
  A10 = policy_improve(usable_ace=True, accuracy=acc, pure_rand=False)
  A00 = policy_improve(usable_ace=False, accuracy=acc, pure_rand=False)
  logging.info(f"{A00}")
  logging.info(f"{A01}")
  logging.info(f"{A10}")
  logging.info(f"{A11}")
  visualize_improve([A00, A01, A10, A11])



def visualize_improve(data: list[np.ndarray]) -> None:
  A00, A01, A10, A11 = data

  fig1  = plt.figure(1, frameon=False)
  ax1 = fig1.add_subplot(1, 1, 1)
  label_axe_improve(fig1, ax1, A00, policy_titles[0])
  fig1.tight_layout()

  fig2  = plt.figure(2, frameon=False)
  ax2 = fig2.add_subplot(1, 1, 1)
  label_axe_improve(fig2, ax2, A01, policy_titles[1])
  fig2.tight_layout()

  fig3  = plt.figure(3, frameon=False)
  ax3 = fig3.add_subplot(1, 1, 1)
  label_axe_improve(fig3, ax3, A10, policy_titles[2])
  fig3.tight_layout()

  fig4  = plt.figure(4, frameon=False)
  ax4 = fig4.add_subplot(1, 1, 1)
  label_axe_improve(fig4, ax4, A11, policy_titles[3])
  fig4.tight_layout()

  plt.show()
  fig1.savefig("./images/A00.png")
  fig2.savefig("./images/A01.png")
  fig3.savefig("./images/A10.png")
  fig4.savefig("./images/A11.png")




def compare(args):
  acc = args.accuracy
  A_book = np.zeros((2, 10, 10), dtype=np.int8)
  A_algo = np.zeros((2, 10, 10), dtype=np.int8)
  ## book
  # usable ace
  A_book[1, :7,  :] = 1
  A_book[1, 6, 2:8] = 1
  # no usable ace
  A_book[0, :5, 6:] = 1
  A_book[0, :5,  0] = 1
  A_book[0, 0, 1:3] = 1
  ## algorithme
  # usable ace
  A_algo[1, :7,  :] = 1
  A_algo[1, 6, 2:8] = 1
  # no usable ace
  A_algo[0, :5, 6:] = 1
  A_algo[0, :5,  0] = 1
  A_algo[0, 0, 1:3] = 1

  policy_test(A_book[1], usable_ace=True, accuracy=acc, pure_rand=True)
  policy_test(A_book[0], usable_ace=False, accuracy=acc, pure_rand=True)
  policy_test(A_algo[1], usable_ace=True, accuracy=acc, pure_rand=True)
  policy_test(A_algo[0], usable_ace=False, accuracy=acc, pure_rand=True)


def main():
  args = parser.parse_args()
  if args.eval:
    evaluation(args)
  else:
    improve(args)

if __name__ == '__main__':
  main()
