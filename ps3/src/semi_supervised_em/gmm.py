import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    n, d = x.shape
    indices = np.array(range(n))
    np.random.seed(100+trial_num)
    np.random.shuffle(indices, )
    mu = np.zeros((K, d))
    sigma = np.zeros((K, d, d))
    batch_size = len(x) // (K - 1)
    for b in range(K):
        
        group = x[indices[batch_size*b: batch_size*(b+1)]]
        mu[b] = group.mean(axis=0)
        # print(f"g: {group.T @ group / n}")
        # print(f"mu: {np.outer(mu[b], mu[b])}")
        sigma[b] = group.T @ group / n - np.outer(mu[b], mu[b])

    phi = np.ones(K) / K

    w = np.ones(K) / K * np.ones((n, K))


    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])
            # print(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n, d).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (d,).
        sigma: Initial cluster covariances, list of k arrays of shape (d, d).

    Returns:
        Updated weight matrix of shape (n, d) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    n, d = x.shape

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        print(f"ll: {ll}")
        # print(f"w: {w}")
        # print(f"phi: {phi}")
        # print(f"mu: {mu}")
        # print(f"sigma: {sigma}")
        # E
        for i in range(n):
            
            exp_v = np.vstack([-(x[i] - mu[j_]) @ np.linalg.inv(sigma[j_]) @ (x[i] - mu[j_]).T / 2 for j_ in range(K)])
            exp_v = exp_v - np.max(exp_v, axis=0)
            # print(f"exp_v {exp_v}")

            denominator = 0.
            for j_ in range(K):
                # print(((x[i] - mu[j_]) @ np.linalg.inv(sigma[j_])@(x[i] - mu[j_]).T))
                denominator += 1. / (np.abs(np.linalg.det(sigma[j_]))**0.5) * np.exp(exp_v[j_]) * phi[j_]

            for j in range(K):
                # print(np.linalg.inv(sigma[j]))
                # print(f"exponented {exp_v[j]}")
                numenator = 1. / (np.abs(np.linalg.det(sigma[j]))**0.5) * np.exp(exp_v[j]) * phi[j]
                
                w[i, j] = numenator / denominator
            # print(f"denominator: {denominator}")
     
        # M
        phi = np.mean(w, axis=0)

        mu = (w.T @ x) / w.sum(axis=0).reshape((K, 1))
        print(f"w: {w}")
        print(f"phi: {phi}")
        print(f"mu: {mu}")

        for j in range(K):
            # phi[j] = np.mean(w[j], axis=0)

            # mu[j] = (w[:, j] @ x) / w[:, j].sum(axis=0)
            # print(np.outer(x[0] - mu[j], x[0] - mu[j]))
            # print(np.sum([w[i, j] * np.outer(x[i] - mu[j], x[i] - mu[j]) for i in range(n)], axis=0))
            sigma[j] = (np.sum([w[i, j] * np.outer(x[i] - mu[j], x[i] - mu[j]) for i in range(n)], axis=0)) / w[:, j].sum(axis=0)
            # print(sigma[j])
            
            # print(w[:, j].sum(axis=0))
            # print(sigma[j] )

        prev_ll = ll
        ll = 0.
        for i in range(n):

            a = (sum(1. / ((2*np.pi)**(d/2) * (np.abs(np.linalg.det(sigma[j]))**0.5)) * np.exp( -(x[i] - mu[j])@np.linalg.inv(sigma[j])@(x[i] - mu[j]).T / 2) * phi[j] for j in range(d)))
            if a > 0.:

                ll += np.log(a)
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n, d).
        x_tilde: Design matrix of labeled examples of shape (n_tilde, d).
        z_tilde: Array of labels of shape (n_tilde, 1).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (d,).
        sigma: Initial cluster covariances, list of k arrays of shape (d, d).

    Returns:
        Updated weight matrix of shape (n, d) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000
    n, d = x.shape
    n_tilde, _ = x_tilde.shape

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***
    
        print(f"ll: {ll}")
        # print(f"w: {w}")
        # print(f"phi: {phi}")
        # print(f"mu: {mu}")
        # print(f"sigma: {sigma}")
        # E
        for i in range(n):
            
            exp_v = np.vstack([-(x[i] - mu[j_]) @ np.linalg.inv(sigma[j_]) @ (x[i] - mu[j_]).T / 2 for j_ in range(K)])
            exp_v = exp_v - np.max(exp_v, axis=0)
            # print(f"exp_v {exp_v}")

            denominator = 0.
            for j_ in range(K):
                # print(((x[i] - mu[j_]) @ np.linalg.inv(sigma[j_])@(x[i] - mu[j_]).T))
                denominator += 1. / (np.abs(np.linalg.det(sigma[j_]))**0.5) * np.exp(exp_v[j_]) * phi[j_]

            for j in range(K):
                # print(np.linalg.inv(sigma[j]))
                # print(f"exponented {exp_v[j]}")
                numenator = 1. / (np.abs(np.linalg.det(sigma[j]))**0.5) * np.exp(exp_v[j]) * phi[j]
                
                w[i, j] = numenator / denominator
            # print(f"denominator: {denominator}")
     

        for j in range(K):
            phi[j] = (np.sum(w[j], axis=0) + alpha*sum(z_tilde==j))/(n + alpha*n_tilde)

            mu[j] = ((w[:, j] @ x) + alpha*(x_tilde*(z_tilde==j)).sum(axis=0)) / (w[:, j].sum(axis=0) + alpha*n_tilde)
            
            sigma[j] = np.sum([w[i, j] * np.outer(x[i] - mu[j], x[i] - mu[j]) for i in range(n)], axis=0)
            

            sigma[j] += alpha*np.sum([int(z_tilde[i]==j)*np.outer(x_tilde[i] - mu[j], x_tilde[i] - mu[j]) 
                                      for i in range(n_tilde)], axis=0)
            
            sigma[j] /= (w[:, j].sum(axis=0) + alpha*sum(z_tilde==j))

        prev_ll = ll
        ll = 0.
        for i in range(n):

            a = (sum(1. / ((2*np.pi)**(d/2) * (np.abs(np.linalg.det(sigma[j]))**0.5)) * np.exp( -(x[i] - mu[j])@np.linalg.inv(sigma[j])@(x[i] - mu[j]).T / 2) * phi[j] for j in range(d)))
            if a > 0.:

                ll += np.log(a)

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # *** END CODE HERE ***
