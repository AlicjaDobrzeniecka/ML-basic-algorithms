import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearModel:

    def __init__(self, theta=None):
        self.theta = theta

    def fit(self, x, y):
        """Fits the linear model using linalg.solve
        Inputs -> x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features)
        Outputs -> y: np.ndarray, dtype=np.float64, shape=(n_examples,)
        Returns: Nothing
        """

        x_transposed = x.T.dot(x)
        y_dot_product = x.T.dot(y)
        self.theta = np.linalg.solve(x_transposed, y_dot_product)

    def predict(self, x):
        """ Makes a prediction given a new set of input features.
        Inputs -> x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features)
        Returns: np.float64
        """
        y_pred = x.dot(self.theta)
        return y_pred


def create_poly(k, x):
    """ Generates polynomial features of the input data x.
    Inputs -> k: degree of polynomial, dtype=int
    Inputs -> x: np.ndarray, dtype=np.float64
    Returns: np.ndarray, dtype=np.float64, shape=(n_examples, k+1)
    """

    poly = PolynomialFeatures(degree=k)
    return poly.fit_transform(x)


def run_model(train_path, p_degree=None, filename='plot.png'):
    if p_degree is None:
        p_degree = [1, 2, 3, 5, 10, 20]

    train_x = np.loadtxt(train_path, delimiter=',', skiprows=1, usecols=[0])
    train_y = np.loadtxt(train_path, delimiter=',', skiprows=1, usecols=[1])

    if train_x.ndim == 1:
        train_x = np.expand_dims(train_x, -1)

    plot_x = np.ones([1000, 1])
    plot_x[:, 0] = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    plt.figure()
    plt.scatter(train_x, train_y)

    for k in p_degree:
        lm_model = LinearModel()
        train_x_poly = create_poly(k, train_x)
        lm_model.fit(train_x_poly, train_y)
        plot_x_poly = create_poly(k, plot_x)
        plot_y = lm_model.predict(plot_x_poly)

        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 0], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path):
    run_model(train_path,  [1, 2, 3, 5, 10, 20], 'polynomials.png')


if __name__ == '__main__':
    main(train_path='train_data.csv')
