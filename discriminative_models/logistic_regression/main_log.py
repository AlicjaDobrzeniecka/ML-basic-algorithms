""" with Newton's method as solver """
from utils import *


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver."""

    def __init__(self, eps=1e-5, theta_0=None):
        self.theta = theta_0
        self.eps = eps

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.
        Inputs => x: Training example inputs. Shape (n_examples, dim).
        Outputs => y: Training example labels. Shape (n_examples,).
        """

        theta_0 = np.zeros(x.shape[1])

        def sigmoid(x_value):
            return 1 / (1 + np.exp(-x_value))

        def cost_function(x_value, y_value, theta):
            predicted_y = sigmoid(x_value.dot(theta))
            cost = (1 / (np.size(x_value, 0))) * (x_value.T.dot((predicted_y - y_value)))
            return cost

        def hessian(x_value, y_value, theta):
            m_hessian = np.zeros((x_value.shape[1], x_value.shape[1]))
            z = y_value * x_value.dot(theta)
            for i in range(m_hessian.shape[0]):
                for j in range(m_hessian.shape[0]):
                    if i <= j:
                        m_hessian[i][j] = np.mean(sigmoid(z) * (1 - sigmoid(z)) * x_value[:, i] * x_value[:, j])
                        if i != j:
                            m_hessian[j][i] = m_hessian[i][j]
            return m_hessian

        theta_f = theta_0

        theta_diff = 1
        while theta_diff > self.eps:
            last_theta = theta_f.copy()
            hessian_inversed = np.linalg.inv(hessian(x, y, theta_f))
            grad = cost_function(x, y, theta_f)
            theta_f = theta_f - hessian_inversed.dot(grad)
            theta_diff = np.linalg.norm(theta_f - last_theta)
        self.theta = theta_f
        return theta_f

    def predict(self, x):
        """Return predicted probabilities given new inputs x
        Inputs => x: of shape (n_examples, dim)
        Returns => binary values of shape (n_examples,)
        """

        def sigmoid(x_value):
            return 1 / (1 + np.exp(-x_value))

        y_prediction = sigmoid(x.dot(self.theta))
        return y_prediction


def run_model(train_path, evaluation_path):
    train_x, train_y = load_dataset(train_path, True)
    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    x_eval, y_eval = load_dataset(evaluation_path, True)
    plot(x_eval, y_eval, clf.theta, "logistic_regression.png")
    p_eval = clf.predict(x_eval)
    y_hat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean((y_hat == 1) == (y_eval == 1)))


if __name__ == '__main__':
    run_model(train_path='train_data.csv', evaluation_path='eval_data.csv')
