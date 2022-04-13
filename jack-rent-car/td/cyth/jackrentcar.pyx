import os
import sys
import numpy as np
cimport numpy as np

cdef int rent_cost = 10
cdef int move_cost = 2
cdef float discount = 0.9
cdef float alpha = 1e-2
cdef int max_garage_cars = 20
cdef int max_moved_car = 5
cdef bint stable = False
cdef list policies = []
cdef float accuracy = 1e-6
cdef float eps = 0.05

cdef list lambda_return = [3, 2]
cdef list lambda_rent = [3, 4]

cdef np.ndarray _A = np.zeros((max_garage_cars + 1, max_garage_cars + 1), dtype=np.int32)
cdef int[:, :] A = _A
cdef np.ndarray _Q = np.zeros((11, max_garage_cars + 1, max_garage_cars + 1), dtype=np.float32)
cdef float[:, :, :] Q = _Q
cdef str root_dir = os.path.dirname(sys.argv[0])


cdef class CarRent:
  cdef float alpha
  def __init__(self):
    self.alpha = 0.1

  cpdef float policy_evalue(self, repeat:int=10000) except -128:
    """
    Evaluate the policy A
    """
    cdef int si, sj, a, ii, jj, rent_i, rent_j, ret_i, ret_j, new_si, new_sj, kk
    cdef np.ndarray Qold = _Q.copy()
    for kk in range(repeat):
      si = np.random.randint(0, 21)
      sj = np.random.randint(0, 21)
      a = self.choose_action(si, sj)
      if a > si:
        a = si
      elif a < 0 and -a > sj:
        a = -sj
      ii = int(si - a)
      jj = int(sj + a)
      ii = min(ii, max_garage_cars)
      jj = min(jj, max_garage_cars)
      rent_i = np.random.poisson(lambda_rent[0])
      rent_j = np.random.poisson(lambda_rent[1])
      rent_i = min(ii, rent_i)
      rent_j = min(jj, rent_j)
      ret_i = np.random.poisson(lambda_return[0])
      ret_j = np.random.poisson(lambda_return[1])
      new_si = min(ii - rent_i + ret_i, max_garage_cars)
      new_sj = min(jj - rent_j + ret_j, max_garage_cars)
      Q[a + 5, si, sj] += \
          self.alpha * ( \
            rent_cost * rent_i \
            - move_cost * abs(a) \
            + discount * np.max(Q[:, new_si, new_sj])  \
            - Q[a + 5, si, sj] \
          )
    return float(np.max(np.abs(_Q - Qold)))


  cpdef int choose_action(self, int s1, int s2) except -128:
    """
    Choose action using epsilon-greedy
    """
    cdef int action = A[s1, s2]
    if np.random.random() >= (1-eps):
      action = np.random.randint(-5, 6)
    return action




  cpdef int policy_improve(self) except -1:
    """
    Sweep over the state space and compute the new best policy
    """
    cdef int i, j
    for i in range(max_garage_cars + 1):
      for j in range(max_garage_cars + 1):
        A[i, j] = np.argmax(Q[:, i, j]) - 5
    return 0



cpdef int jackrentcar() except -1:
  np.set_printoptions(linewidth=100000)
  cdef CarRent car_rent = CarRent()
  cdef float delta
  cdef i = 0
  while True:
    print("Evaluate Policies ...")
    delta = car_rent.policy_evalue()
    if delta < 0.1:
      print("Evaluate Finished !")
      break
    print("Improve Policies ...")
    car_rent.policy_improve()
    i+=1
    if i & 7 == 0:
      print(_A)
  return 0
