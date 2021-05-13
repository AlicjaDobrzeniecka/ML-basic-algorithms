from utils import *
import matplotlib.pyplot as plt


class PoissonRegression:
    def __init__(self, step_size=1e-5, eps=1e-5, theta_0=None):
        self.theta = theta_0
        self.step_size = step_size
        self.eps = eps

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Inputs => x: shape (n_examples, dim)
        Inputs => y: shape (n_examples,)
        """
        theta_0 = np.zeros((x.shape[1]))
        diff = 1
        theta_f = theta_0
        while diff > self.eps:
            last_theta = theta_f.copy()
            h = np.exp(x.dot(theta_f))
            change = self.step_size * (y - h).dot(x)
            theta_f += change
            diff = np.linalg.norm((theta_f - last_theta), ord=1)
        self.theta = theta_f
        return self.theta

    def predict(self, x):
        """Make a prediction given inputs x.
        Inputs => x: of shape (n_examples, dim).
        Returns: floats, shape (n_examples,).
        """

        return np.exp(x.dot(self.theta))


def run_model(lr, train_path, eval_path):
    train_x, train_y = load_dataset(train_path, True)
    clf = PoissonRegression(step_size=lr)
    clf.fit(train_x, train_y)
    x_eval, y_eval = load_dataset(eval_path, True)
    p_eval = clf.predict(x_eval)
    plt.figure()
    plt.scatter(y_eval,p_eval,alpha=0.4,c='red',label='Ground Truth vs Predicted')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig('poisson_valid.png')


if __name__ == '__main__':
    run_model(lr=1e-5, train_path='train.csv', eval_path='valid.csv')
