import logging
import os
import sys
import numpy as np

from time import time
from scipy.special import factorial

rent_cost = 10
move_cost = 2
discount = 0.9
alpha = 1e-2
max_garage_cars = 20
max_moved_car = 5
stable = False
policies = []
accuracy = 1e-6
eps = 0.05

lambda_return = [3, 2]
lambda_rent = [3, 4]

A = np.zeros((max_garage_cars + 1, max_garage_cars + 1), dtype=np.int32)
Q = np.zeros((11, max_garage_cars + 1, max_garage_cars + 1), dtype=np.float32)
root_dir = os.path.dirname(sys.argv[0])


def poisson_prob(lam, n):
   return np.exp(-lam) * (lam**n) / factorial(n)



class CarRent:
  def __init__(self):
    self.alpha = 0.1
  def policy_evalue(self, repeat:int=10000) -> float:
    """
    Evaluate the policy A
    """
    Qold = Q.copy()
    for _ in range(repeat):
      si: int = np.random.randint(0, 21)
      sj: int = np.random.randint(0, 21)
      a: int = self.choose_action(si, sj)
      if a > si:
        a = si
      elif a < 0 and -a > sj:
        a = -sj
      ii: int = int(si - a)
      jj: int = int(sj + a)
      ii = min(ii, max_garage_cars)
      jj = min(jj, max_garage_cars)
      rent_i: int = np.random.poisson(lambda_rent[0])
      rent_j: int = np.random.poisson(lambda_rent[1])
      rent_i = min(ii, rent_i)
      rent_j = min(jj, rent_j)
      ret_i = np.random.poisson(lambda_return[0])
      ret_j = np.random.poisson(lambda_return[1])
      new_si = min(ii - rent_i + ret_i, max_garage_cars)
      new_sj = min(jj - rent_j + ret_j, max_garage_cars)
      alpha = self.alpha
      Q[a + 5, si, sj] += \
          alpha * ( \
            + rent_cost * rent_i \
            - move_cost * abs(a) \
            + discount * (np.argmax(Q[:, new_si, new_sj]) - 5)  \
            - Q[a + 5, si, sj] \
          )
    return float(np.max(np.abs(Q - Qold)))


  def choose_action(self, s1: int, s2: int) -> int:
    """
    Choose action using epsilon-greedy
    """
    action: int = A[s1, s2]
    if np.random.random() >= (1-eps):
      action = np.random.randint(-5, 6)
    return action




  def policy_improve(self):
    """
    Sweep over the state space and compute the new best policy
    """
    for i in range(max_garage_cars + 1):
      for j in range(max_garage_cars + 1):
        A[i, j] = np.argmax(Q[:, i, j]) - 5



def main():
  np.set_printoptions(linewidth=100000)
  car_rent = CarRent()
  while True:
    print("Evaluate Policies...")
    delta = car_rent.policy_evalue()
    if delta < 0.1:
      print("Evaluate Finished!")
      break
    print("Improve Policies...")
    car_rent.policy_improve()
    print(Q)
    policies.append(A.copy())

if __name__ == "__main__":
  main()
