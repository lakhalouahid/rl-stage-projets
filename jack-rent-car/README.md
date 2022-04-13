## Problème de Location de voitures de Jack

### Description du problème
Voir le livre "Reinforcement learning: An Introduction", Chapitre 4, page: 81.


### Les Résultats

#### Politique 1
```
[[ 0  0  0  0  0  0  0  0 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -3 -4 -4]
 [ 0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3]
 [ 0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2 -2 -2 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1]
 [ 1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1]
 [ 2  2  2  1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  3  2  2  2  2  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  3  3  3  3  2  2  2  1  1  1  1  0  0  0  0  0  0  0  0  0]
 [ 4  4  4  4  3  3  3  2  2  2  2  1  1  1  0  0  0  0  0  0  0]
 [ 5  5  5  4  4  4  3  3  3  3  2  2  2  1  1  1  0  0  0  0  0]
 [ 5  5  5  5  5  4  4  4  4  3  3  3  2  2  2  1  1  1  0  0  0]
 [ 5  5  5  5  5  5  5  5  4  4  4  3  3  3  2  2  2  1  1  0  0]
 [ 5  5  5  5  5  5  5  5  5  5  4  4  4  3  3  3  2  2  1  0  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  4  4  4  3  3  2  1  1  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  4  3  2  2  1  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  3  2  1  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  4  3  2  1  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1  0]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1  0]]
```


#### Politique 2
```
[[ 0  0  0  0  0  0  0 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -3 -4 -4 -4]
 [ 0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3 -3]
 [ 0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -1]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1]
 [ 1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  3  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  3  2  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  4  3  2  2  2  1  1  1  1  1  1  1  1  0  0  0  0  0]]

```


#### Politique 3
```
[[ 0  0  0  0  0  0  0 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -4 -4 -4 -4]
 [ 0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -3]
 [ 0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -2 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -1 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1]
 [ 1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  3  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  3  2  1  1  1  1  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  4  3  2  2  2  2  1  1  1  1  1  0  0  0  0  0  0]
 [ 5  5  5  5  4  3  3  3  3  2  2  2  2  2  1  1  1  0  0  0  0]]

```


#### Politique 4
```
[[ 0  0  0  0  0  0  0 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -4 -4 -4 -4]
 [ 0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -3]
 [ 0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -2 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -1 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1]
 [ 1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  3  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  2  1  1  1  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  3  2  2  2  1  1  1  1  1  0  0  0  0  0  0  0]
 [ 5  5  5  4  4  3  3  3  2  2  2  2  2  1  1  1  1  0  0  0  0]]

```


#### Politique 5
```
[[ 0  0  0  0  0  0  0 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -4 -4 -4 -4]
 [ 0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -3 -3 -3 -3 -3]
 [ 0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -2 -2 -2 -2 -2 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -1 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1]
 [ 1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  3  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  2  1  1  1  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  3  2  2  2  1  1  1  1  1  0  0  0  0  0  0  0]
 [ 5  5  5  4  4  3  3  3  2  2  2  2  2  1  1  1  1  0  0  0  0]]

```