import gym
import numpy as np

env_name = "MountainCar-v0"
env = gym.make(env_name)
obs = env.reset()
env.render()

# Some initializations
#
n_states = 40
episodes = 5000
initial_lr = 1.0
min_lr = 0.005
gamma = 0.99
epsilon = 0.05

env = env.unwrapped
env.seed()
np.random.seed(0)


# Quantize the states
#
def discretization(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_den = (env_high - env_low) / n_states
    pos_den, vel_den = env_den
    pos_low, vel_low = env_low
    pos_scaled = int((obs[0] - pos_low) / pos_den)
    vel_scaled = int((obs[1] - vel_low) / vel_den)
    return pos_scaled, vel_scaled


q_table = np.zeros((n_states, n_states, env.action_space.n))
print(q_table.shape)

#
for episode in range(episodes):
    print("Episode:", episode)
    obs = env.reset()
    alpha = max(min_lr, initial_lr * (gamma ** (episode // 100)))
    while True:
        if episode >= (episodes - 1):
            env.render()
        # env.render()
        pos, vel = discretization(env, obs)
        if np.random.uniform(low=0, high=1) < epsilon:
            a = np.random.choice(env.action_space.n)
        else:
            a = np.argmax(q_table[pos][vel])
        obs, reward, terminate, _ = env.step(a)
        pos_, vel_ = discretization(env, obs)

        # Q function update
        q_table[pos][vel][a] = (1 - alpha) * q_table[pos][vel][a] + alpha * (
                reward + gamma * np.max(q_table[pos_][vel_]))

        if terminate:
            break

env.close()
with open('q_table.npy', 'wb') as f:
    np.save(f, q_table)

