import logging
import numpy as np

from datetime import datetime
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

logger_name = "jack-rent-car"
logger_filename = f"logs/{logger_name}-{datetime.now().isoformat()}.log"
formatter = logging.Formatter(fmt="%(message)s")

file_handler = logging.FileHandler(logger_filename)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger = logging.getLogger(name=logger_name)
logger.addHandler(file_handler)


def poisson_prob(lam, n):
   return np.exp(-lam) * (lam**n) / factorial(n)


class CarRent:
  def trans_prob(self, s, garage):
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
    for i in range(max_garage_cars+1):
      self.trans_prob(i, 0)
      self.trans_prob(i, 1)

  def policy_evalue(self):
    delta = 0
    for i in range(max_garage_cars+1):
      for j in range(max_garage_cars+1):
        v = V[i, j]
        a = A[i, j]
        V[i, j] = self.value_calculate(i, j, a)
        delta = max(delta, np.abs(v-V[i, j]))
    return delta

  def value_calculate(self, i, j, a):
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
    stable_flag = True
    for i in range(max_garage_cars + 1):
      for j in range(max_garage_cars + 1):
        act_best = self.action_greedy(i, j)
        if act_best != A[i, j]:
          A[i, j] = act_best
          stable_flag = False
    return stable_flag


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


if __name__ == "__main__":
  main()
