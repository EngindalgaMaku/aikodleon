import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Pekiştirmeli Öğrenme | Python Veri Bilimi | Kodleon',
  description: 'Python kullanarak pekiştirmeli öğrenme temellerini, algoritmaları ve uygulamaları öğrenin.',
};

const content = `
# Python ile Pekiştirmeli Öğrenme

Pekiştirmeli öğrenme, bir ajanın çevresiyle etkileşime girerek deneme-yanılma yoluyla öğrenmesini sağlayan bir makine öğrenmesi yaklaşımıdır. Bu bölümde, Python ile pekiştirmeli öğrenme uygulamalarını öğreneceğiz.

## Temel Kavramlar

### Ortam ve Ajan

\`\`\`python
import gym
import numpy as np

# OpenAI Gym ortamı oluşturma
env = gym.make('CartPole-v1')

# Ortam bilgilerini görüntüleme
print("Eylem uzayı:", env.action_space)
print("Durum uzayı:", env.observation_space)

# Basit bir rastgele ajan
for episode in range(3):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Rastgele bir eylem seç
        action = env.action_space.sample()
        
        # Eylemi uygula
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        state = next_state
    
    print(f"Bölüm {episode + 1}: Toplam ödül = {total_reward}")
\`\`\`

### Q-Öğrenme

\`\`\`python
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-değeri güncelleme
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value

# Örnek kullanım
env = gym.make('FrozenLake-v1')
agent = QLearningAgent(16, 4)  # 16 durum, 4 eylem

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.learn(state, action, reward, next_state)
        state = next_state

print("Q-tablosu:\\n", agent.q_table)
\`\`\`

## Deep Q-Network (DQN)

### DQN Uygulaması

\`\`\`python
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Örnek kullanım
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

EPISODES = 100
batch_size = 32

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            print(f"Bölüm: {e}/{EPISODES}, skor: {time}")
            break
            
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
\`\`\`

## Policy Gradient

### REINFORCE Algoritması

\`\`\`python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.gamma = 0.99
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def update(self, rewards, log_probs):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

# Örnek kullanım
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = REINFORCEAgent(state_size, action_size)

for episode in range(500):
    state = env.reset()
    rewards = []
    log_probs = []
    
    for t in range(1000):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        rewards.append(reward)
        log_probs.append(log_prob)
        
        state = next_state
        
        if done:
            break
    
    agent.update(rewards, log_probs)
    print(f"Bölüm {episode}: Toplam ödül = {sum(rewards)}")
\`\`\`

## Actor-Critic

### A2C Uygulaması

\`\`\`python
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.gamma = 0.99
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), value
    
    def update(self, rewards, log_probs, values, next_value):
        returns = []
        R = next_value
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        values = torch.cat(values)
        
        advantage = returns - values
        
        actor_loss = []
        for log_prob, adv in zip(log_probs, advantage):
            actor_loss.append(-log_prob * adv.detach())
        
        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = advantage.pow(2).mean()
        
        total_loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

# Örnek kullanım
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = A2CAgent(state_size, action_size)

for episode in range(500):
    state = env.reset()
    rewards = []
    log_probs = []
    values = []
    
    for t in range(1000):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        
        state = next_state
        
        if done:
            break
    
    _, _, next_value = agent.select_action(next_state)
    agent.update(rewards, log_probs, values, next_value)
    print(f"Bölüm {episode}: Toplam ödül = {sum(rewards)}")
\`\`\`

## Alıştırmalar

1. **Temel Pekiştirmeli Öğrenme**
   - Farklı OpenAI Gym ortamlarını deneyin
   - Q-öğrenme algoritmasını farklı problemlere uygulayın
   - Epsilon-greedy stratejisini değiştirin

2. **Deep Q-Learning**
   - DQN mimarisini özelleştirin
   - Deneyim tekrarı mekanizmasını geliştirin
   - Double DQN ve Dueling DQN implementasyonları yapın

3. **Policy Gradient ve Actor-Critic**
   - REINFORCE algoritmasını farklı ortamlarda test edin
   - A2C algoritmasını özelleştirin
   - PPO (Proximal Policy Optimization) implementasyonu yapın

## Sonraki Adımlar

1. [Büyük Veri](/topics/python/veri-bilimi/buyuk-veri)
2. [Yapay Zeka Projeleri](/topics/python/veri-bilimi/yapay-zeka-projeleri)
3. [İleri Seviye Makine Öğrenmesi](/topics/python/veri-bilimi/ileri-makine-ogrenmesi)

## Faydalı Kaynaklar

- [OpenAI Gym Dokümantasyonu](https://gym.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
`;

export default function ReinforcementLearningPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Veri Bilimi
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 