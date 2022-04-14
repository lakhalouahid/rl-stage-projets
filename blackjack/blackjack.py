import numpy as np
from numba import jit, boolean, int8, int32, int64
from numba.experimental import jitclass

np.set_printoptions(linewidth=10000, precision=5)



cards_space = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=np.int8)
stick_action: int = 0
push_action: int = 1

@jit(nopython=True, cache=True)
def sample_card() -> int:
  return np.random.choice(cards_space)




@jitclass([('count', int32), ('shoe', int8[:]), ('pure_rand', boolean)])
class Shoe():
  count: int
  shoe: np.ndarray
  pure_rand: bool

  def __init__(self, count: int, shoe: np.ndarray, pure_rand: bool):
    self.count = count
    self.shoe = shoe
    self.pure_rand = pure_rand

  def sample_card(self):
    if self.pure_rand:
      return np.random.choice(cards_space)
    card = self.shoe[self.count]
    if self.count == 0:
      self.count = self.shoe.shape[0]
      self.shoe = self.shoe[np.random.permutation(self.count)]
    self.count -= 1
    return card

@jit(nopython=True, cache=True)
def get_sum(cards: np.ndarray, usable_ace: bool) -> int:
  has_ace = False
  cards_sum = 0
  for card in cards:
    cards_sum += card
    if card == 1:
      has_ace == True
  if usable_ace and has_ace and cards_sum + 10 <= 21:
      return cards_sum + 10
  return cards_sum

@jit(nopython=True, cache=True)
def dealer_policy(dealer_cards: np.ndarray, player_cards: np.ndarray, usable_ace: bool) -> int:
  player_sum = get_sum(player_cards, usable_ace)
  if player_sum > 21:
    return stick_action
  dealer_sum = get_sum(dealer_cards, usable_ace)
  if dealer_sum < 17:
    return push_action
  return stick_action

@jit(nopython=True, cache=True)
def player_policy(dealer_cards: np.ndarray, player_cards: np.ndarray, usable_ace: bool, A: np.ndarray) -> int:
  player_sum = get_sum(player_cards, usable_ace)
  if player_sum <= 21:
    return A[player_sum - 12, dealer_cards[1] - 1]
  return stick_action


@jit(nopython=True, cache=True)
def get_result(dealer_cards: np.ndarray, player_cards: np.ndarray, usable_ace: bool) -> int:
  player_sum = get_sum(player_cards, usable_ace)
  dealer_sum = get_sum(dealer_cards, usable_ace)
  if player_sum > 21:
    return -1
  elif dealer_sum > 21:
    return 1
  elif player_sum < dealer_sum:
    return -1
  elif player_sum > dealer_sum:
    return 1
  return 0

@jit(nopython=True, cache=True)
def history_player_cards(player_cards: np.ndarray, i :int):
  return player_cards[np.arange(i+2)]

@jit(nopython=True, cache=True)
def value_update(dealer_cards: np.ndarray, player_cards: np.ndarray, result: int, V: np.ndarray, nV: np.ndarray, usable_ace: bool) -> None:
  player_sum = get_sum(player_cards, usable_ace)
  for i in range(len(player_cards)-1):
    if player_cards[i+1] == 0:
      break
    player_sum = get_sum(history_player_cards(player_cards, i), usable_ace=usable_ace)
    if player_sum <= 21 and player_sum >= 12:
      ii = player_sum - 12
      jj = dealer_cards[1] - 1
      V[ii, jj] += (1.0 / nV[ii, jj]) * (result - V[ii, jj])
      nV[ii,  jj] += 1

@jit(nopython=True, cache=True)
def append_card(cards: np.ndarray, i: int, card: int) -> int:
  cards[i] = card
  return i+1

@jit(nopython=True, cache=True)
def clear_cards(cards: np.ndarray) -> None:
  cards.fill(0)


@jit(nopython=True, cache=True)
def policy_eval(
    usable_ace: bool=False,
    accuracy: float=1e-2,
    pure_rand: bool=True) -> tuple[np.ndarray, np.ndarray]:
  player_cards = np.zeros((21,), dtype=np.int8)
  dealer_cards = np.zeros((21,), dtype=np.int8)
  A = np.zeros((10, 10), dtype=np.int8)
  V = np.zeros((10, 10), dtype=np.float32)
  nV = np.ones((10, 10), dtype=np.int32)
  shoe_buffer = np.repeat(cards_space, 4 * 8)
  count = shoe_buffer.shape[0] - 1
  shoe = Shoe(count, shoe_buffer, pure_rand)
  A[:5, :] = 1
  while True:
    V_old = V.copy()
    for _ in range(1000):
      ii = 0
      jj = 0
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())

      while get_sum(player_cards, usable_ace) < 12:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while player_policy(dealer_cards, player_cards, usable_ace, A) == push_action:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while dealer_policy(dealer_cards, player_cards, usable_ace) == push_action:
        jj = append_card(dealer_cards, jj, shoe.sample_card())

      result = get_result(dealer_cards, player_cards, usable_ace)

      value_update(dealer_cards, player_cards, result, V, nV, usable_ace)

      clear_cards(player_cards)
      clear_cards(dealer_cards)
    delta = np.max(np.abs(V - V_old))
    if delta < accuracy:
      break
  print("Evaluation finished!")
  return V, nV
