import numpy as np
import matplotlib.pyplot as plt


def load_dataset(data_path: str, intercept: bool):
    train_x = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=[0, 1])
    train_y = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=[2])

    if train_x.ndim == 1:
        train_x = np.expand_dims(train_x, -1)

    if intercept:
        """Add intercept to matrix x => new matrix same as x with 1's in the 0th column """
        new_x = np.zeros((train_x.shape[0], train_x.shape[1] + 1), dtype=train_x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = train_x
        train_x = new_x.copy()

    return train_x, train_y


def plot(x, y, theta, save_path, correction=1.0):
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)
