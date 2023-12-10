import numpy as np
import util
import time
from tqdm import tqdm


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    print(clf.theta)
    util.plot(x_eval, y_eval, clf.theta, f'{save_path.split(".")[0]}-plot')
    y_pred = clf.predict(x_eval)
    np.savetxt(save_path, y_pred) 
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
        10000
    """
    def __init__(self, step_size=0.01, max_iter=1000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(self.dim + 1)
        
        for i in tqdm(range(self.max_iter), disable=not self.verbose):

            dJ = sum(x[j]*(y[j] - self.sigmoid(self.theta, x[j]))
                    for j in range(len(x)))
            
            J = sum(y[j]*np.log(self.sigmoid(self.theta, x[j])) + \
                (1-y[j])*np.log(1-self.sigmoid(self.theta, x[j]))
                for j in range(len(x)))

            dtheta = self.step_size * dJ / J 
            self.theta = self.theta - dtheta
            if np.linalg.norm(dtheta) < self.eps:
                break


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
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
