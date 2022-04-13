import logging
import os
import sys
import numpy as np

from time import time
from matplotlib import cm, pyplot as plt
from scipy.special import factorial

rent_cost = 10
move_cost = 2
discount = 0.9
max_garage_cars = 20
max_moved_car = 5
stable = False
policies = []
accuracy = 1e-6

lambda_return = [3, 2]
lambda_rent = [3, 4]

A = np.zeros((21,21), dtype=np.int32)
V = np.zeros((21,21), dtype=np.float32)
R = np.zeros((2 ,21), dtype=np.float32)
T = np.zeros((2 ,21,21), dtype=np.float32)
root_dir = os.path.dirname(sys.argv[0])
logger_name = "jack-rent-car"
logger_filename = f"{root_dir}/logs/{logger_name}-{int(time())}.log"
formatter = logging.Formatter("%(message)s")

file_handler = logging.FileHandler(logger_filename)
file_handler.setFormatter(formatter)

logger = logging.getLogger(logger_name)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def poisson_prob(lam, n):
   return np.exp(-lam) * (lam**n) / factorial(n)


class CarRent:
  def trans_prob(self, s, garage):
    """
    Update the transition probalility matrix starting from stat 's' for the location 'garage'

    this function loops first with all possible renting
    requests then loops over all possible returns, then
    it updated the matrices
    """
    for r in range(max_garage_cars + 1):
      p_rent = poisson_prob(lambda_rent[garage], r)
      if p_rent < accuracy:
        return
      rent = min(s, r)
      R[garage, s] += p_rent * rent_cost * rent
      for ret in range(max_garage_cars + 1):
        p_ret = poisson_prob(lambda_return[garage], ret)
        if p_ret < accuracy:
          continue
        s_next = min(s - rent + ret, max_garage_cars)
        T[garage, s, s_next] += p_rent * p_ret

  def init_trans_prob(self):
    """
    Calculate the transition probability maxtrix
    """
    for i in range(max_garage_cars+1):
      self.trans_prob(i, 0)
      self.trans_prob(i, 1)

  def policy_evalue(self):
    """
    Evaluate the policy A
    """
    delta = 0
    for i in range(max_garage_cars+1):
      for j in range(max_garage_cars+1):
        v = V[i, j]
        a = A[i, j]
        V[i, j] = self.value_calculate(i, j, a)
        delta = max(delta, np.abs(v - V[i, j]))
    return delta

  def value_calculate(self, i, j, a):
    """
    Compute the value of a action given a state
    """
    if a > i:
      a = i
    elif a < 0 and -a > j:
      a = -j
    ii = int(i - a)
    jj = int(j + a)
    ii = min(ii, max_garage_cars)
    jj = min(jj, max_garage_cars)
    temp_v = -np.abs(a) * move_cost
    for m in range(max_garage_cars + 1):
      for n in range(max_garage_cars + 1):
        temp_v += T[0,ii,m]*T[1,jj,n]*(R[0,ii] + R[1,jj] + discount*V[m,n])
    return temp_v

  def action_greedy(self, i, j):
    """
    Compute the greedy action given a state
    """
    best_action = 0
    best_value = 0
    for a in range(-max_moved_car, max_moved_car+1):
      if a > i:
        continue
      elif a < 0 and -a > j:
        continue
      val = self.value_calculate(i, j, a)
      if val > (best_value + 0.1):
        best_value = val
        best_action = a
    return best_action

  def policy_improve(self):
    """
    Sweep over the state space and compute the new best policy
    """
    stable_flag = True
    for i in range(max_garage_cars + 1):
      for j in range(max_garage_cars + 1):
        act_best = self.action_greedy(i, j)
        if act_best != A[i, j]:
          A[i, j] = act_best
          stable_flag = False
    return stable_flag



def visualize_results():
  fig = plt.figure()
  ax1 = fig.add_subplot(2, 3, 1)
  fig.pad_inches = -1
  img1 = ax1.imshow(policies[0])
  ax1.set_title("Policy plot after 1 iteration")
  ax1.set_xlabel("Number of cars in station A")
  ax1.set_ylabel("Number of cars in station B")
  plt.colorbar(img1)
  ax2 = fig.add_subplot(2, 3, 2)
  ax2.imshow(policies[1])
  ax2.set_title("Policy plot after 2 iterations")
  ax2.axis('off')
  ax3 = fig.add_subplot(2, 3, 3)
  ax3.set_title("Policy plot after 3 iterations")
  ax3.imshow(policies[2])
  ax3.axis('off')
  ax4 = fig.add_subplot(2, 3, 4)
  ax4.set_title("Policy plot after 4 iterations")
  ax4.imshow(policies[3])
  ax4.axis('off')
  ax5 = fig.add_subplot(2, 3, 5)
  ax5.set_title("Policy plot after 5 iterations")
  ax5.imshow(policies[4])
  ax5.axis('off')
  ax6 = fig.add_subplot(2, 3, 6, projection='3d')
  X = np.arange(21)
  Y = np.arange(21)
  X, Y = np.meshgrid(X, Y)
  surf = ax6.contourf(X, Y, V, 250, cmap=cm.coolwarm, alpha=0.6, antialiased=False)
  fig.colorbar(surf, ax=ax6)
  ax6.set_title("Optimal value Plot")
  ax6.set_xlabel("Number of cars in station A")
  ax6.set_ylabel("Number of cars in station B")
  ax6.set_zlabel("State value")
  fig.tight_layout()
  plt.show()

def main():
  np.set_printoptions(linewidth=100000)
  car_rent = CarRent()
  car_rent.init_trans_prob()
  stable = False
  while not stable:
    print("Evaluate Policies...")
    while 1:
      delta = car_rent.policy_evalue()
      if delta < 0.1:
        print("Evaluate Finished!")
        break
    print("Improve Policies...")
    stable = car_rent.policy_improve()
    logger.info(A)
    policies.append(A.copy())
  visualize_results()

if __name__ == "__main__":
  main()
