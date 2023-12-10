import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    clf = GDA()
    clf.fit(x_train, y_train)
    print(clf.theta)
    util.plot(x_eval, y_eval, clf.theta, f'{save_path.split(".")[0]}-plot')
    y_pred = clf.predict(x_eval)
    np.savetxt(save_path, y_pred) 
    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.dim = 2
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        n = len(x)
        phi = sum(y == 1) / n
        mu_0 = sum(x[i][1:] * (y[i] == 0) for i in range(n)) / sum(y == 0)
        mu_1 = sum(x[i][1:] * (y[i] == 1) for i in range(n)) / sum(y == 1)
        mu = {
            0: mu_0,
            1: mu_1

        }
        sigma = sum((x[i][1:] - mu[y[i]]).reshape(-1, 1) @ \
            (x[i][1:] - mu[y[i]]).reshape(-1, 1).T for i in range(n)) / n
        sigma_inv = np.linalg.inv(sigma)  
        self.theta = np.zeros(self.dim + 1)
        self.theta[0] = np.log(phi / (1 - phi)) + \
            1/2*(mu_0.reshape(-1, 1).T @ sigma_inv @ mu_0.reshape(-1, 1) - \
                mu_1.reshape(-1, 1).T @ sigma_inv @ mu_1.reshape(-1, 1))
        self.theta[1:] = (mu_1 - mu_0).reshape(-1, 1).T @ sigma_inv    
        # *** END CODE HERE ***

    @staticmethod
    def sigmoid(theta, x):
        return 1 / (1 + np.exp(-np.dot(theta, x)) + 1e-8)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        preds = np.array([])
        for i in range(len(x)):
            pred = self.sigmoid(self.theta, x[i]) > 0.5
            preds = np.append(preds, pred)

        return preds
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
