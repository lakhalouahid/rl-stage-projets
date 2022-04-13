import numpy as np
np.set_printoptions(linewidth=10000, precision=3)

usable_ace = True
accuracy = 1e-4
cards_space = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
cards_space = np.array(cards_space, dtype=np.int32)


push_action, stick_action = 1, 0
A = np.zeros((10, 10), dtype=np.int8)
V = np.zeros((10, 10), dtype=np.float32)
nV = np.ones((10, 10), dtype=np.int32)
R = np.array([-1, 0, 1], dtype=np.int32)

A[:8, :] = 1
print(A)

def sample_card():
  return np.random.choice(cards_space)

def get_sum(cards):
  has_ace = False
  cards_sum = 0
  for card in cards:
    cards_sum += card
    if card == 1:
      has_ace == True
  if usable_ace and has_ace and cards_sum + 10 <= 21:
      return cards_sum + 10
  else:
    return cards_sum

def dealer_policy(dealer_cards, player_cards):
  player_sum = get_sum(player_cards)
  if player_sum > 21:
    return stick_action
  else:
    dealer_sum = get_sum(dealer_cards)
    if dealer_sum < 17:
      return push_action
    else:
      return stick_action

def player_policy(dealer_cards, player_cards):
  player_sum = get_sum(player_cards)
  if player_sum <= 21:
    dealer_showed_card = dealer_cards[1]
    return A[player_sum - 12, dealer_showed_card - 1]
  else:
    return stick_action


def get_result(dealer_cards, player_cards):
  player_sum = get_sum(player_cards)
  dealer_sum = get_sum(dealer_cards)
  if player_sum > 21:
    return -1
  elif dealer_sum > 21:
    return 1
  elif player_sum < dealer_sum:
    return -1
  elif player_sum > dealer_sum:
    return 1
  elif player_sum == dealer_sum:
    return 0

def value_update(dealer_cards, player_cards, result):
  player_sum = get_sum(player_cards)
  dealer_showed_card = dealer_cards[1]
  for i in range(len(player_cards)-1):
    player_sum = get_sum([player_cards[j] for j in range(0, i+2)])
    if player_sum <= 21:
      n = nV[player_sum - 12, dealer_showed_card - 1]
      V[player_sum - 12,  dealer_showed_card - 1] += (1./n) * (result - V[player_sum - 12, dealer_showed_card - 1])
      nV[player_sum - 12, dealer_showed_card - 1] += 1


def policy_eval():
  player_cards, dealer_cards = [], []
  i = 0
  while True:
    i += 1
    prev_V = V.copy()
    for _ in range(1000):
      player_cards.append(sample_card())
      dealer_cards.append(sample_card())
      player_cards.append(sample_card())
      dealer_cards.append(sample_card())

      while get_sum(player_cards) < 12:
        player_cards.append(sample_card())

      while player_policy(dealer_cards, player_cards) == push_action:
        player_cards.append(sample_card())

      while dealer_policy(dealer_cards, player_cards) == push_action:
        dealer_cards.append(sample_card())
      result = get_result(dealer_cards, player_cards)

      value_update(dealer_cards, player_cards, result)

      player_cards.clear()
      dealer_cards.clear()

    delta = np.max(abs(V - prev_V))
    print(i)
    if delta < accuracy:
      print("Evaluation finished!")
      print(V)
      print(nV)
      break

policy_eval()
