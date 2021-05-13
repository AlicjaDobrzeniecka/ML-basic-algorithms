from utils import *


class GDA:
    """Gaussian Discriminant Analysis"""

    def __init__(self, step_size=0.01, eps=1e-5, theta_0=None):
        self.theta = theta_0
        self.step_size = step_size
        self.eps = eps

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating self.theta.
        Inputs => x: of shape (n_examples, dim)
        Outputs => y: of shape (n_examples,)"""

        self.theta = np.zeros((3,))

        where_y_is_one = (y == 1.0)
        mu_1 = x[where_y_is_one].sum(axis=0) / len(x[where_y_is_one])
        mu_0 = x[~where_y_is_one].sum(axis=0) / len(x[~where_y_is_one])
        phi = len(x[where_y_is_one]) / len(x)
        sigma = ((x[where_y_is_one] - mu_1).T.dot(x[where_y_is_one] - mu_1) + (x[~where_y_is_one] - mu_0).T.dot(
            (x[~where_y_is_one] - mu_0))) / len(x)
        sigma_inversed = np.linalg.pinv(sigma)
        theta_1 = -np.dot((mu_0 - mu_1).T, sigma_inversed)
        theta_0 = -(-0.5 * (np.dot(mu_0, sigma_inversed).dot(mu_0) - np.dot(mu_1, sigma_inversed).dot(mu_1)) - np.log(
            (1.0 - phi) / phi))

        self.theta[0] = theta_0
        self.theta[1] = theta_1[0]
        self.theta[2] = theta_1[1]

    def predict(self, x):
        """Make a prediction given new inputs x
        Inputs => x: of shape (n_examples, dim)
        Returns: shape (n_examples,)"""
        predictions = []
        for row_x in x:
            theta_x = self.theta.transpose().dot(row_x)
            sigmoid = 1.0 / (1.0 + np.exp(-theta_x))
            predictions.append(sigmoid)
        return np.array(predictions)


def run_model(train_path, valid_path):
    """Problem: Gaussian discriminant analysis (GDA)"""
    x_train, y_train = load_dataset(train_path, False)
    clf = GDA()
    clf.fit(x_train, y_train)
    x_eval, y_eval = load_dataset(valid_path, True)
    plot(x_eval, y_eval, clf.theta, "gaussian.png")
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('GDA Accuracy: %.2f' % np.mean((yhat == 1) == (y_eval == 1)))


if __name__ == '__main__':
    run_model(train_path='train_data.csv', valid_path='valid_data.csv')
