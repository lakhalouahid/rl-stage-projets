# Le jeu cartpole

## Description du problème

Un pôle est fixé à une année une-titres attachés à un chariot, qui se déplace
le long d'une piste sans friction. Le système est contrôlé en appliquant une
résistance de +1 ou -1 au chariot. Le pendule commence de droit et le but est
de l'empêcher de tomber. Une récompense de +1 est fournie pour chaque horaire
que le pôle reste debout. L'épisode se termine lorsque le pôle est supérieur à
15 degrés de vertical, de l'or Le panier déplace plus de 2,4 unités du centre.

## Version 1

### Implémentation

#### Replay Memory

```python
class carray():
  def __init__(self, maxlen, size, dtype):
    if isinstance(size, tuple):
      shape = (maxlen, *size)
    else:
      shape = (maxlen, size)
    self.buf = np.zeros(shape=shape, dtype=dtype)
    self.maxlen = maxlen
    self.head = 0
    self.len = 0

  def append(self, item):
    self.buf[self.head] = item
    if self.len < self.maxlen:
      self.len += 1
    if self.head == 0:
      self.head = self.maxlen
    self.head -= 1

  def sample_from_perm(self, perm):
    perm = perm + self.head + 1
    perm = perm & (self.maxlen - 1)
    return self.buf[perm]

class Buffer():
  def __init__(self, maxlen: int):
    self.states = carray(maxlen, 4, np.float32)
    self.actions = carray(maxlen, 1, np.int8)
    self.newstates = carray(maxlen, 4, np.float32)
    self.rewards = carray(maxlen, 1, np.int8)
    self.dones = carray(maxlen, 1, np.bool8)
    self.maxlen = maxlen
    self.len = 0

  def push(self, experience):
    state, action, newstate, reward, done = experience
    self.states.append(state)
    self.actions.append(action)
    self.newstates.append(newstate)
    self.rewards.append(reward)
    self.dones.append(done)
    if self.len < self.maxlen:
      self.len += 1

  def sample(self, batch_size):
    perm = np.random.permutation(self.len)[:batch_size]
    states = self.states.sample_from_perm(perm)
    actions = self.actions.sample_from_perm(perm)
    newstates = self.newstates.sample_from_perm(perm)
    rewards = self.rewards.sample_from_perm(perm)
    dones = self.dones.sample_from_perm(perm)
    return (states, actions, newstates, rewards, dones)
```

#### DQNN

```python
class DQNN(nn.Module):
  def __init__(self, nntype, discount=1, lr=3e-4, lr_decay=0.999, min_lr=3e-6, itermax=60):
    super(DQNN, self).__init__()
    self.train_net = nntype()
    self.target_net = nntype()
    self.target_itermax = itermax
    self.save_freq = 0
    self.target_iter = 0
    self.discount = discount
    self.min_lr = min_lr
    self.optimizer = Adam(params=self.train_net.parameters(), lr=lr, weight_decay=0, amsgrad=False)
    self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_decay)

  def forward(self, x):
    x = torch.from_numpy(x)
    return self.train_net(x)

  def forward_target(self, x):
    x = torch.from_numpy(x)
    return self.target_net(x)


  def train_dqn(self, batch, steps_nmbr):
    states, actions, newstates, rewards, dones = batch
    actions = torch.from_numpy(actions)
    rewards = torch.from_numpy(rewards)
    dones = torch.from_numpy(dones)
    with torch.no_grad():
      q_newstates = self.forward_target(newstates)
    maxq_newstates = torch.max(q_newstates, dim=1).values.reshape(-1, 1)
    maxq_newstates = (1 - torch.as_tensor(dones, dtype=torch.int32)) *  maxq_newstates
    for param in self.parameters():
      param.grad = None
    q_states = self(states)
    loss = torch.mean((self.discount * maxq_newstates + rewards - \
        torch.take_along_dim(q_states, torch.as_tensor(actions, dtype=torch.int64), dim=1))**2)
    loss.backward()
    self.optimizer.step()
    self.target_switch(steps_nmbr)
    if self.scheduler.get_lr()[0] > self.min_lr:
      self.scheduler.step()

  def eval_dqn(self, x):
    with torch.no_grad():
      return int(torch.argmax(self.forward_target(x)))

  def target_switch(self, steps_nmbr):
    self.target_iter += 1
    if self.target_iter >= self.target_itermax:
      self.target_net.load_state_dict(self.train_net.state_dict())
      self.target_iter = 0
      self.save_freq += 1
      if self.save_freq & 3 == 0:
        torch.save(self.state_dict(), f"checkpoints/deep-q-model-{steps_nmbr}-{int(time()):5d}.pt")
  def load_model(self, path):
    self.load_state_dict(torch.load(path))
```

#### Boucle d'entraînement

```python
def train(lr, lr_decay, min_lr, itermax):
  try:
    eps = 0.05
    env = gym.make('CartPole-v1')
    buffer = Buffer(1<<16)
    # dqnn = DQNN(NN, lr=lr, lr_decay=lr_decay, min_lr=min_lr, itermax=itermax)
    for i_episode in range(int(1e5)):
      state = env.reset()
      for t in range(500):
        action = 0
        if i_episode & 1023 == 0:
          env.render(mode='human')
          action = dqnn.eval_dqn(state)
        else:
          actions_values = dqnn(state)
          action = choose_action(actions_values, eps)
        newstate, reward, done, info = env.step(action)
        buffer.push((state, action, newstate, reward, done))
        state = newstate
        if done:
          dqnn.train_dqn(buffer.sample(1<<8), t+1)
          if i_episode & 7 == 0:
            if i_episode & 255 == 0:
              print(f"Test episode {i_episode} finished after {t+1} steps for {itermax}")
            else:
              print(f"Episode {i_episode} finished after {t+1} steps for {itermax}")
          break

    env.close()
  except KeyboardInterrupt as e:
    print("exit")
```


#### Boucle de test


```python
def test():
  try:
    env = gym.make('CartPole-v1')
    dqnn = DQNN(NN)
    dqnn.load_model('checkpoints/deep-q-model-500-1650182128.pt') # modéle à utiliser
    for _ in range(int(1e5)):
      state = env.reset()
      for t in range(500):
        env.render()
        action = dqnn.eval_dqn(state)
        newstate, reward, done, info = env.step(action)
        state = newstate
        if done:
          break

    env.close()
  except KeyboardInterrupt as e:
    print("exit")
```



#### Résultats:

![vidéo d'entraìnement](./videos/video_.mp4)


## Version 2
