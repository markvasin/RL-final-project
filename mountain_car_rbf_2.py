import gym
import numpy as np
import sklearn
from sklearn.cluster import KMeans

env_name = "MountainCar-v0"
env = gym.make(env_name)
obs = env.reset()
env.render()

# Some initializations
#
n_actions = env.action_space.n
n_states = 40
episodes = 500
initial_lr = 1.0
min_lr = 0.005
gamma = 0.99
epsilon = 0.05

env = env.unwrapped
env.seed()
np.random.seed(0)

J = 100
sig = 1.0

observation_examples = []
for x in range(10000):
    tmp = env.observation_space.sample().tolist()
    tmp.append(np.random.choice(env.action_space.n))
    observation_examples.append(tmp)

observation_examples = np.array(observation_examples)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)
scaler.transform(observation_examples)
kmeans = KMeans(n_clusters=J, random_state=0).fit(scaler.transform(observation_examples))
cluster_centers = kmeans.cluster_centers_


def gaussian(x, u, sigma):
    return np.exp(-0.5 * np.linalg.norm(x - u) / sigma)


# Construct design matrix
def construct_design_matrix(X):
    X = scaler.transform(X)
    N = X.shape[0]
    U = np.zeros((N, J))
    for i in range(N):
        for j in range(J):
            U[i][j] = gaussian(X[i], cluster_centers[j], sig)

    return U


def q(p, v, a, w):
    U = get_features(p, v, a)
    yh = U @ w
    return yh


def get_features(p, v, a):
    features = np.array([[p, v, a]])
    U = construct_design_matrix(features)
    return U


def get_action(p, v, w):
    states = np.array([[p, v, action] for action in range(n_actions)])
    U = construct_design_matrix(states)
    yh = U @ w
    if np.random.uniform(low=0, high=1) < epsilon:
        chosen_action = np.random.choice(env.action_space.n)
    else:
        chosen_action = np.argmax(yh)
    return chosen_action


w = np.zeros(J)

for episode in range(episodes):
    print("Episode:", episode)
    obs = env.reset()
    alpha = max(min_lr, initial_lr * (gamma ** (episode // 100)))
    action = get_action(obs[0], obs[1], w)
    step = 0
    while True:
        if episode >= (episodes - 1):
            env.render()
        # env.render()
        curr_pos, curr_vel = obs
        curr_action = action
        obs, reward, terminate, _ = env.step(action)
        new_pos, new_vel = obs
        action = get_action(new_pos, new_vel, w)
        feat = get_features(curr_pos, curr_vel, curr_action)
        w += alpha * (reward + gamma * q(new_pos, new_vel, action, w) - q(curr_pos, curr_vel, curr_action,
                                                                          w)) * feat.flatten()
        if terminate:
            break

        step += 1

env.close()
