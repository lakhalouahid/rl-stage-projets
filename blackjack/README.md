# Le jeu Blackjack (simple version)

## Description du problème

L'objet du jeu de cartes de casino populaire de Blackjack est d'obtenir des
cartes la somme des valeurs numériques aussi grandes que possible sans dépasser
21. Toutes les cartes de visage comptent comme 10, et une ACE peut compter en 1
ou comme 11. Nous considérons que nous considérons La version dans laquelle
chaque joueur est en concurrence indépendamment contre le concessionnaire. Le
jeu commence par deux cartes traitées au concessionnaire et au joueur. L'une
des cartes du revendeur est face visible et l'autre est face à face. Si le
joueur a 21 immédiatement (un ACE et une 10 cartes), il est appelé naturel. Il
gagne ensuite à moins que le revendeur ait également un naturel, auquel cas le
jeu est un tirage au sort. Si le joueur n'a pas de naturel, il peut alors
demander des cartes supplémentaires, une par une (hits), jusqu'à ce qu'il
arrête (bâtonnets) ou dépasse 21 (passe le buste). S'il fait faillite, il perd;
S'il collasse, il devient le tour du revendeur. Le concessionnaire frappe ou
colle selon une stratégie fixe sans choix: il colle sur une somme de 17 ou
plus, et frappe autrement. Si le courtier fait faillite, le joueur gagne;
Sinon, la victoire de résultat, perdre ou dessiner - est déterminée par la
somme finale de laquelle la somme finale est plus proche de 21. La lecture du
blackjack est naturellement formulée comme un PMD fini épisodique. Chaque jeu
de Blackjack est un épisode. Les récompenses de +1, -1 et 0 sont données pour
gagner, perdre et dessiner, respectivement. Toutes les récompenses dans un jeu
sont zéro et nous ne discounts pas (γ = 1); Par conséquent, ces récompenses du
terminal sont également les retours. Les actions du joueur sont à frapper ou à
coller. Les états dépendent des cartes du joueur et de la carte de montage du
revendeur. Nous supposons que les cartes sont traitées d'un pont infini
(c'est-à-dire avec remplacement) afin qu'il n'y ait aucun avantage à garder une
trace des cartes déjà traitées. Si le joueur détient un ACE qu'il pouvait
compter comme 11 sans faire de buste, l'Ace est dit être utilisable. Dans ce
cas, il est toujours compté comme 11 parce que le comptant comme 1 rendrait la
somme 11 ou moins, auquel cas il n'y a aucune décision à faire car, évidemment,
le joueur devrait toujours frapper. Ainsi, le joueur prend des décisions sur la
base de trois variables: sa somme actuelle (12-21), la carte d'affichage du
concessionnaire (ACE-10) et si elle détient ou non un ACE utilisable. Cela fait
pour un total de 200 états.


## Implémentation

### Utilisation du numba, pour JIT (Just In Time) Compilation

Pour plus d'informations, visitez [numba](https://numba.pydata.org/)


```python
from numba import jit, boolean, int8, int16
from numba.experimental import jitclass # jit a class
```


### class Shoe

```python
@jitclass([('count', int16), ('shoe', int8[:]), ('pure_rand', boolean)])
class Shoe():
  count: int # current card index
  shoe: np.ndarray # shoe of cards
  pure_rand: bool # if pure_rand == True => sample without replacement

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
      self.shoe = self.shoe[np.random.permutation(self.count)] # shuffle the shoe
    self.count -= 1
    return card
```

### Calcul du somme des cartes en prise en compte activation du "ace"

```python
@jit('i1(i1[:],b1)', nopython=True, cache=True, fastmath=True)
def get_sum(cards: np.ndarray, usable_ace: bool) -> int:
  has_ace = False
  cards_sum = 0
  for i in np.arange(cards.shape[0]):
    cards_sum += cards[i]
    if cards[i] == 1:
      has_ace = True
  if usable_ace and has_ace and (cards_sum + 10) <= 21:
    return cards_sum + 10
  return cards_sum
```

### Implémentation de la politique du "dealer"

la politique du dealer prise en compte l'activation du "ace"

```python
@jit('i1(i1[:],i1[:],b1)', nopython=True, cache=True, fastmath=True)
def dealer_policy(dealer_cards: np.ndarray, player_cards: np.ndarray, usable_ace: bool) -> int:
  player_sum = get_sum(player_cards, usable_ace)
  if player_sum > 21:
    return stick_action
  dealer_sum = get_sum(dealer_cards, usable_ace)
  if dealer_sum < 17:
    return push_action
  return stick_action
```


### Implémentation de la politique du "player" en mode eval

la politique du player prise en compte l'activation du "ace"

```python
@jit('i1(i1[:],i1[:],i1[:,:],b1)', nopython=True, cache=True, fastmath=True)
def player_policy(
    dealer_cards: np.ndarray,
    player_cards: np.ndarray,
    A: np.ndarray,
    usable_ace: bool) -> int:
  player_sum = get_sum(player_cards, usable_ace)
  if player_sum <= 21 and player_sum >= 12:
    return A[player_sum - 12, dealer_cards[1] - 1]
  return stick_action
```


### Implémentation de la politique du "player" en mode estimation

la politique du player prise en compte l'activation du "ace"

```python
@jit('i1(i1,f4)', nopython=True, cache=True, fastmath=True)
def select_eps_greedy_action(action: int, eps: float=0.05) -> int:
  if np.random.random() < (1-eps):
    return action
  return np.random.randint(2)

@jit('i1(i1[:],i1[:],i1[:,:],b1,f4)', nopython=True, cache=True, fastmath=True)
def player_policy_estimation(
    dealer_cards: np.ndarray,
    player_cards: np.ndarray,
    A: np.ndarray,
    usable_ace: bool,
    eps: float) -> int:
  player_sum = get_sum(player_cards, usable_ace)
  if player_sum <= 21 and player_sum >= 12:
    greedy_action = A[player_sum - 12, dealer_cards[1] - 1]
    return select_eps_greedy_action(greedy_action, eps)
  return stick_action
```

### Calcul du résultat

```python
@jit('i1(i1[:],i1[:],b1)', nopython=True, cache=True, fastmath=True)
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
```

### Mise à jour du value state en mode eval

```python
@jit('void(i1[:],i1[:],f4[:,:],i4[:,:],i1,b1)', nopython=True, cache=True, fastmath=True)
def value_update(
    dealer_cards: np.ndarray,
    player_cards: np.ndarray,
    V: np.ndarray,
    nV: np.ndarray,
    result: int,
    usable_ace: bool) -> None:
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
```


### Mise à jour du value state en mode estimation

```python
@jit('void(i1[:],i1[:],f4[:,:,:],i4[:,:,:],i1,b1)', nopython=True, cache=True, fastmath=True)
def value_update_estimation(
    dealer_cards: np.ndarray,
    player_cards: np.ndarray,
    Q: np.ndarray,
    nQ: np.ndarray,
    result: int,
    usable_ace: bool) -> None:
  player_sum = get_sum(player_cards, usable_ace)
  player_sum = get_sum(player_cards, usable_ace)
  for i in range(len(player_cards)-1):
    if player_cards[i+1] == 0:
      break
    player_sum = get_sum(history_player_cards(player_cards, i), usable_ace=usable_ace)
    if player_sum <= 21 and player_sum >= 12:
      aa = ((i+2) != player_cards[player_cards > 0].shape[0]) * 1
      ii = player_sum - 12
      jj = dealer_cards[1] - 1
      Q[ii, jj, aa] += (1.0 / nQ[ii, jj, aa]) * (result - Q[ii, jj, aa])
      nQ[ii,  jj, aa] += 1
```

### Mise à jour de la politique


```python
@jit('void(i1[:,:],f4[:,:,:])', nopython=True, cache=True, fastmath=True)
def update_policy(A: np.ndarray, Q: np.ndarray) -> None:
  for ii in range(A.shape[0]):
    for jj in range(A.shape[1]):
      A[ii, jj] = np.argmax(Q[ii, jj])
```


### Évaluation d'une politique

```python
@jit(nopython=True, cache=True)
def policy_eval(
    A: np.ndarray,
    usable_ace: bool=False,
    accuracy: float=1e-2,
    pure_rand: bool=True) -> tuple[np.ndarray, np.ndarray]:
  player_cards = np.zeros((21,), dtype=np.int8)
  dealer_cards = np.zeros((21,), dtype=np.int8)
  V = np.zeros((10, 10), dtype=np.float32)
  nV = np.ones((10, 10), dtype=np.int32)
  shoe_buffer = np.repeat(cards_space, 4 * 8)
  count = shoe_buffer.shape[0] - 1
  shoe = Shoe(count, shoe_buffer, pure_rand)
  while True:
    V_old = V.copy()
    for _ in range(5000):
      ii = 0
      jj = 0
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())

      while get_sum(player_cards, usable_ace) < 12:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while player_policy(dealer_cards, player_cards, A, usable_ace) == push_action:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while dealer_policy(dealer_cards, player_cards, usable_ace) == push_action:
        jj = append_card(dealer_cards, jj, shoe.sample_card())

      result = get_result(dealer_cards, player_cards, usable_ace)

      value_update(dealer_cards, player_cards, V, nV, result, usable_ace)

      clear_cards(player_cards)
      clear_cards(dealer_cards)

    if np.max(np.abs(V - V_old)) < accuracy:
      break
  print("Evaluation finished!")
  return V, nV

```


### Trouver la meilleure politique

```python
@jit(nopython=True, cache=True)
def policy_improve(
    usable_ace: bool=False,
    accuracy: float=1e-2,
    pure_rand: bool=True,
    eps: float=0.05) -> np.ndarray:
  player_cards = np.zeros((21,), dtype=np.int8)
  dealer_cards = np.zeros((21,), dtype=np.int8)
  A = np.zeros((10, 10), dtype=np.int8)
  Q = np.zeros((10, 10, 2), dtype=np.float32)
  nQ = np.ones((10, 10, 2), dtype=np.int32)
  shoe_buffer = np.repeat(cards_space, 4 * 8)
  count = shoe_buffer.shape[0] - 1
  shoe = Shoe(count, shoe_buffer, pure_rand)
  A[:8, :] = 1
  while True:
    Q_old = Q.copy()
    for _ in range(5000):
      ii = 0
      jj = 0
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())

      while get_sum(player_cards, usable_ace) < 12:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while player_policy_estimation(dealer_cards, player_cards, A, usable_ace, eps) == push_action:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while dealer_policy(dealer_cards, player_cards, usable_ace) == push_action:
        jj = append_card(dealer_cards, jj, shoe.sample_card())

      result = get_result(dealer_cards, player_cards, usable_ace)

      value_update_estimation(dealer_cards, player_cards, Q, nQ, result, usable_ace)


      clear_cards(player_cards)
      clear_cards(dealer_cards)

    update_policy(A, Q)

    if np.max(np.abs(Q - Q_old)) < accuracy:
      break
  print("Evaluation finished!")
  return A
```

### Tester une politique


```python
@jit(nopython=True, cache=True)
def policy_test(
    A: np.ndarray,
    usable_ace: bool=False,
    accuracy: float=1e-2,
    pure_rand: bool=True) -> tuple[float, float]:
  player_cards = np.zeros((21,), dtype=np.int8)
  dealer_cards = np.zeros((21,), dtype=np.int8)
  V = np.zeros((10, 10), dtype=np.float32)
  nV = np.ones((10, 10), dtype=np.int32)
  shoe_buffer = np.repeat(cards_space, 4 * 8)
  count = shoe_buffer.shape[0] - 1
  shoe = Shoe(count, shoe_buffer, pure_rand)
  sim_i = 0.0
  win_i = 0.0
  drw_i = 0.0
  while True:
    V_old = V.copy()
    for _ in range(5000):
      sim_i += 1
      ii = 0
      jj = 0
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())
      ii = append_card(player_cards, ii, shoe.sample_card())
      jj = append_card(dealer_cards, jj, shoe.sample_card())

      while get_sum(player_cards, usable_ace) < 12:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while player_policy(dealer_cards, player_cards, A, usable_ace) == push_action:
        ii = append_card(player_cards, ii, shoe.sample_card())

      while dealer_policy(dealer_cards, player_cards, usable_ace) == push_action:
        jj = append_card(dealer_cards, jj, shoe.sample_card())

      result = get_result(dealer_cards, player_cards, usable_ace)
      if result == 1:
        win_i += 1
      elif result == 0:
        drw_i += 1

      value_update(dealer_cards, player_cards, V, nV, result, usable_ace)

      clear_cards(player_cards)
      clear_cards(dealer_cards)

    if np.max(np.abs(V - V_old)) < accuracy:
      break
  print("Evaluation finished!")
  return (win_i / sim_i, drw_i / sim_i)
```

## Résultats

### Évaluation de la politique trivial

#### Description:

Pour évaluer une politique, on fixe les paramétres suivants:

1. `accuracy`
2. `usable_ace` Vrai si l'ace est activité
3. `pure_rand` Vrai si le tirage est sans remise

**La politique** : "Tant que la somme <= 19, prend nouvelle carte"

##### Résultats d'évaluation

###### Cas d'ace désactivé et tirage avec remise

[image](./images/V00.png)


###### Cas d'ace désactivé et tirage sans remise

[image](./images/V01.png)


###### Cas d'ace activé et tirage avec remise

[image](./images/V10.png)


###### Cas d'ace activé et tirage sans remise

[image](./images/V11.png)
