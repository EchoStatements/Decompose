from decompose import Bernoulli, Categorical, Gaussian, MultivariateGaussian, Poisson
from decompose import BinaryCrossEntropy, CrossEntropy, SquaredLoss, PoissonLoss
import numpy as np

def test_binary_cross_entropy(verbose=False):
    # generate some data for binary cross entropy
    D = 3 # number of runs
    M = 2 # ensemble size
    N = 10 # size of test set
    all_close = True
    np.random.seed(0)
    data = np.random.uniform(0, 1, size=(D, M, N))
    labels = np.random.choice(2, size=(N))
    breg_decomp = BinaryCrossEntropy(data, labels)
    kl_decomp = Bernoulli(data, labels)
    # Ensemble variance
    kl_ensemble_variance = kl_decomp.ensemble_variance
    bregman_ensemble_variance = breg_decomp.ensemble_variance

    if verbose:
        print("Ensemble variance")
        print(kl_ensemble_variance.mean())
        print(bregman_ensemble_variance.mean())

    # Average variance
    kl_average_var = kl_decomp.average_variance
    bregman_average_var = breg_decomp.average_variance
    if verbose:
        print("Average variance")
        print(kl_average_var.mean())
        print(bregman_average_var.mean())

    kl_diversity = kl_decomp.diversity
    bregman_diversity = breg_decomp.diversity
    if verbose:
        print("Diversity")
        print(kl_diversity.mean())
        print(bregman_diversity.mean())

    kl_disparity = kl_decomp.disparity
    bregman_disparity = breg_decomp.disparity

    if verbose:
        print("Disparity")
        print(kl_disparity.mean())
        print(bregman_disparity.mean())

    all_close = all_close and (np.allclose(kl_ensemble_variance, bregman_ensemble_variance))
    all_close = all_close and (np.allclose(kl_average_var, bregman_average_var))
    all_close = all_close and (np.allclose(kl_diversity, bregman_diversity))
    all_close = all_close and (np.allclose(kl_disparity, bregman_disparity))
    print(f"Binary Cross Entropy all close:\t\t {all_close}")

def test_cross_entropy(verbose=False):
    # generate some data for binary cross entropy
    D = 3 # number of runs
    M = 2 # ensemble size
    N = 10 # size of test set
    K = 4 # Num classes
    np.random.seed(0)
    data = np.random.uniform(0, 1, size=(D, M, N, K))
    data /= data.sum(axis=3, keepdims=True)
    labels = np.random.choice(K, size=(N))
    breg_decomp = CrossEntropy(data, labels)
    kl_decomp = Categorical(data, labels)
    # Ensemble variance
    kl_ensemble_variance = kl_decomp.ensemble_variance
    bregman_ensemble_variance = breg_decomp.ensemble_variance
    if verbose:
        print("Ensemble variance")
        print(kl_ensemble_variance.mean())
        print(bregman_ensemble_variance.mean())

    # Average variance
    kl_average_var = kl_decomp.average_variance
    bregman_average_var = breg_decomp.average_variance
    if verbose:
        print("Average variance")
        print(kl_average_var.mean())
        print(bregman_average_var.mean())

    kl_diversity = kl_decomp.diversity
    bregman_diversity = breg_decomp.diversity
    if verbose:
        print("Diversity")
        print(kl_diversity.mean())
        print(bregman_diversity.mean())

    kl_disparity = kl_decomp.disparity
    bregman_disparity = breg_decomp.disparity
    if verbose:
        print("Disparity")
        print(kl_disparity.mean())
        print(bregman_disparity.mean())

    all_close = True
    all_close = all_close and (np.allclose(kl_ensemble_variance, bregman_ensemble_variance))
    all_close = all_close and (np.allclose(kl_average_var, bregman_average_var))
    all_close = all_close and (np.allclose(kl_diversity, bregman_diversity))
    all_close = all_close and (np.allclose(kl_disparity, bregman_disparity))
    print(f"Cross Entropy all close:\t\t {all_close}")

def test_squared_loss(verbose=False):
    # generate some data for binary cross entropy
    D = 3 # number of runs
    M = 2 # ensemble size
    N = 10 # size of test set
    np.random.seed(0)
    data = np.random.normal(0, 1, size=(D, M, N))
    labels = np.random.normal(size=(N))
    breg_decomp = SquaredLoss(data, labels)
    kl_decomp = Gaussian(data, labels)
    # Ensemble variance
    kl_ensemble_variance = kl_decomp.ensemble_variance
    bregman_ensemble_variance = breg_decomp.ensemble_variance
    if verbose:
        print("Ensemble variance")
        print(2 * kl_ensemble_variance.mean())
        print(bregman_ensemble_variance.mean())
        print(f"Are the same: {np.allclose(kl_ensemble_variance, bregman_ensemble_variance)}")

    # Average variance
    kl_average_var = kl_decomp.average_variance
    bregman_average_var = breg_decomp.average_variance
    if verbose:
        print("Average variance")
        print(2 * kl_average_var.mean())
        print(bregman_average_var.mean())

    kl_diversity = kl_decomp.diversity
    bregman_diversity = breg_decomp.diversity
    if verbose:
        print("Diversity")
        print(2 * kl_diversity.mean())
        print(bregman_diversity.mean())

    kl_disparity = kl_decomp.disparity
    bregman_disparity = breg_decomp.disparity
    if verbose:
        print("Disparity")
        print(2 * kl_disparity.mean())
        print(bregman_disparity.mean())

    all_close = True
    all_close = all_close and (np.allclose(2 * kl_ensemble_variance, bregman_ensemble_variance))
    all_close = all_close and (np.allclose(2 * kl_average_var, bregman_average_var))
    all_close = all_close and (np.allclose(2 * kl_diversity, bregman_diversity))
    all_close = all_close and (np.allclose(2 * kl_disparity, bregman_disparity))
    print(f"Squared Loss all close:\t\t {all_close}")


def test_poisson_loss(verbose=False):
    # generate some data for binary cross entropy
    D = 3  # number of runs
    M = 2  # ensemble size
    N = 10  # size of test set
    np.random.seed(0)
    data = np.random.uniform(0, 10, size=(D, M, N))
    labels = np.random.choice(10, size=(N))
    breg_decomp = PoissonLoss(data, labels)
    kl_decomp = Poisson(data, labels)
    # Ensemble variance
    kl_ensemble_variance = kl_decomp.ensemble_variance
    bregman_ensemble_variance = breg_decomp.ensemble_variance
    if verbose:
        print("Ensemble variance")
        print(kl_ensemble_variance.mean())
        print(bregman_ensemble_variance.mean())
        print(f"Are the same: {np.allclose(kl_ensemble_variance, bregman_ensemble_variance)}")

    # Average variance
    kl_average_var = kl_decomp.average_variance
    bregman_average_var = breg_decomp.average_variance
    if verbose:
        print("Average variance")
        print(kl_average_var.mean())
        print(bregman_average_var.mean())
    # print(kl_decomp._KL_difference((), 0, 1).mean())
    # print(breg_decomp._bregman_expectation(0, ()).mean())

    kl_diversity = kl_decomp.diversity
    bregman_diversity = breg_decomp.diversity
    if verbose:
        print("Diversity")
        print(kl_diversity.mean())
        print(bregman_diversity.mean())

    kl_disparity = kl_decomp.disparity
    bregman_disparity = breg_decomp.disparity
    if verbose:
        print("Disparity")
        print(kl_disparity.mean())
        print(bregman_disparity.mean())


    all_close = True
    all_close = all_close and (np.allclose(kl_ensemble_variance, bregman_ensemble_variance))
    all_close = all_close and (np.allclose(kl_average_var, bregman_average_var))
    all_close = all_close and (np.allclose(kl_diversity, bregman_diversity))
    all_close = all_close and (np.allclose(kl_disparity, bregman_disparity))
    print(f"Poisson Loss all close:\t\t {all_close}")


def main():
    verbose = False
    test_binary_cross_entropy(verbose=verbose)
    test_cross_entropy(verbose=verbose)
    test_squared_loss(verbose=verbose)
    test_poisson_loss(verbose=verbose)


if __name__=="__main__":
    main()