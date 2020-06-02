import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def gaussian(x, u, sigma):
    return np.exp(-0.5 * np.linalg.norm(x - u) / sigma)


with open('q_table.npy', 'rb') as f:
    q_table = np.load(f)

n_states = 40
n_actions = 3

data = np.zeros((n_states * n_states * n_actions, 4))

idx = 0
for i in range(n_states):
    for j in range(n_states):
        for k in range(n_actions):
            data[idx] = np.array([i, j, k, q_table[i][j][k]])
            idx += 1

N, pp1 = data.shape

X = np.array(data[:, :-1])
y = np.array(data[:, -1])
print(X.shape, y.shape)

w = np.linalg.inv(X.T @ X) @ X.T @ y
yh_lin = X @ w
# plt.plot(y, yh_lin, '.', Color='magenta')
# plt.show()

J = 20
kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
sig = np.std(X)
print('sigma', sig)

# Construct design matrix
U = np.zeros((N, J))
for i in range(N):
    for j in range(J):
        U[i][j] = gaussian(X[i], kmeans.cluster_centers_[j], sig)

w = np.linalg.inv(U.T @ U) @ U.T @ y

yh_rbf = U @ w

print(y)

plt.plot(y, yh_rbf, '.', Color='cyan')
plt.show()

print(np.linalg.norm(y - yh_lin), np.linalg.norm(y - yh_rbf))

print(w.shape)
