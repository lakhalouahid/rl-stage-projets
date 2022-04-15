import unittest

import numpy as np

from blackjack import get_sum, dealer_policy, stick_action, push_action

class TestBlackJackMethods(unittest.TestCase):

  def test_get_sum(self):
    ## no usable ace
    self.assertEqual(get_sum(np.array([4,  9], dtype=np.int8), usable_ace=False), 13)
    self.assertEqual(get_sum(np.array([1, 10], dtype=np.int8), usable_ace=False), 11)
    self.assertEqual(get_sum(np.array([10, 10], dtype=np.int8), usable_ace=False), 20)
    self.assertEqual(get_sum(np.array([1, 10, 4], dtype=np.int8), usable_ace=False), 15)
    self.assertEqual(get_sum(np.array([1, 1, 10], dtype=np.int8), usable_ace=False), 12)

    ## usable ace
    self.assertEqual(get_sum(np.array([1, 10], dtype=np.int8), usable_ace=True), 21)
    self.assertEqual(get_sum(np.array([1, 1, 10], dtype=np.int8), usable_ace=True), 12)
    self.assertEqual(get_sum(np.array([1, 10, 10], dtype=np.int8), usable_ace=True), 21)
    self.assertEqual(get_sum(np.array([1, 10, 4], dtype=np.int8), usable_ace=True), 15)

  def test_dealer_policy(self):
    self.assertEqual(dealer_policy(
      np.array([4, 10], dtype=np.int8),
      np.array([10, 10], dtype=np.int8),
      False), push_action)
    self.assertEqual(dealer_policy(
      np.array([18, 10], dtype=np.int8),
      np.array([10, 10], dtype=np.int8),
      False), stick_action)
    self.assertEqual(dealer_policy(
      np.array([13, 10], dtype=np.int8),
      np.array([9, 10, 4], dtype=np.int8),
      False), stick_action)


if __name__ == '__main__':
    unittest.main()
