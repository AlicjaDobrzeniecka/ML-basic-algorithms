import numpy as np


def load_dataset(data_path: str, intercept: bool):
    train_x = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=[0, 1, 2, 3])
    train_y = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=[4])

    if train_x.ndim == 1:
        train_x = np.expand_dims(train_x, -1)

    if intercept:
        """Add intercept to matrix x => new matrix same as x with 1's in the 0th column """
        new_x = np.zeros((train_x.shape[0], train_x.shape[1] + 1), dtype=train_x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = train_x
        train_x = new_x.copy()

    return train_x, train_y
