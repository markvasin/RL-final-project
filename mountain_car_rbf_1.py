import gym
import numpy as np
from sklearn.cluster import KMeans


def gaussian(x, u, sigma):
    return np.exp(-0.5 * np.linalg.norm(x - u) / sigma)


with open('q_table.npy', 'rb') as f:
    q_table = np.load(f)

env_name = "MountainCar-v0"
env = gym.make(env_name)
obs = env.reset()
env.render()

n_states = 40
n_actions = env.action_space.n
state_dimension = 3
data = np.zeros((n_states * n_states * n_actions, state_dimension + 1))

idx = 0
for i in range(n_states):
    for j in range(n_states):
        for k in range(n_actions):
            data[idx] = np.array([i, j, k, q_table[i][j][k]])
            idx += 1

N, pp1 = data.shape

X = np.array(data[:, :-1])
y = np.array(data[:, -1])

J = 300
kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
sig = np.std(X)
print('sigma', sig)

# Construct design matrix
U = np.zeros((N, J))
for i in range(N):
    for j in range(J):
        U[i][j] = gaussian(X[i], kmeans.cluster_centers_[j], sig)

w = np.linalg.inv(U.T @ U) @ U.T @ y

yh = U @ w

print('loss', np.linalg.norm(y - yh))
# Some initializations
#
episodes = 10

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


def get_action(p, v):
    states = np.array([[p, v, action] for action in range(n_actions)])
    N, features = states.shape
    U = np.zeros((N, J))
    for i in range(N):
        for j in range(J):
            U[i][j] = gaussian(states[i], kmeans.cluster_centers_[j], sig)

    yh = U @ w
    best_action = np.argmax(yh)
    return best_action


for episode in range(episodes):
    print("Episode:", episode)
    obs = env.reset()
    while True:
        env.render()
        pos, vel = discretization(env, obs)
        a = get_action(pos, vel)
        obs, reward, terminate, _ = env.step(a)
        if terminate:
            break

env.close()