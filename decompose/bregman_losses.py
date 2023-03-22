from decompose import BregmanDecomposition
import numpy as np
from scipy.special import factorial, loggamma, xlogy

class BinaryCrossEntropy(BregmanDecomposition):
    """
    Create a :py:class:`BregmanDecomposition` object given data for binary classification problems.

    Parameters
    ----------
    pred : array_like
        A (D, M, N)-shape array of conditional mean estimates,
        where D is the number of training
        data resamples, M is the number of ensemble members, and
        N is the number of test data points.
    labels : array_like
        An (N,)-shape array of integer labels (0 or 1), where N is the number of
        test data points.

    See Also
    --------
    CrossEntropy
    """

    def __init__(self, pred, labels):
        super().__init__(pred, labels)

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
            'number of test datapoints.')
        if np.any(np.logical_and(labels != 0, labels != 1)):
            raise ValueError('Only labels of 0 and 1 are allowed.')
        if np.any(np.logical_or(pred <= 1e-12, pred >= 1 - 1e-12)):
            raise ValueError('Only predictions between 0 and 1 (not inclusive) allowed')

    def _generator_gradient(self, q=None):
        # The canonical link function for the Bernoulli.
        if q is None:
            return np.log(self.pred) - np.log(1 - self.pred)
        else:
            return np.log(q) - np.log(1 - q)

    @staticmethod
    def _inverse_generator_gradient(etas):
        # The canonical inverse link for the Bernoulli.
        return 1 / (1 + np.exp(-etas))

    def bregman_generator(self, pred):
        """
        Bregman generator for binary cross entropy. The Bregman generator takes scalar values (i.e., only the probability
        of the positive class needs to be given, not a vector containing the probabilities of both classes).

        Parameters
        ----------
        pred : ndarray
            Array of values to which the generator is to abe applied.

        Returns
        -------
        output : ndarray
            Array of same shape as pred

        """
        return pred * np.log(pred) + (1. - pred) * np.log(1. - pred)

    @staticmethod
    def _compute_error(pred, labels):
        label_shape = np.ones_like(pred.shape)
        label_shape[-1] = -1
        labels1 = labels.reshape(label_shape)
        return - labels1 * np.log(pred) - (1 - labels1) * np.log(1 - pred)

class CrossEntropy(BregmanDecomposition):
    """
    Create a :py:class:`BregmanDecomposition` object for multi-class classification problems.

    Parameters
    ----------
    pred : array_like
        A (D, M, N, K)-shape array of conditional mean estimates,
        where D is the number of training
        data resamples, M is the number of ensemble members,
        N is the number of test data points, and K is the number of
        possible outcomes/classes.
    labels : array_like
        An (N,)-shape array of integer labels in :math:`[0, 1, \\ldots, K-1]`, where N is the number of
        test data points.

    See Also
    --------
    BinaryCrossEntropy
    """
    def __init__(self, pred, labels):
        super().__init__(pred, labels)

        if len(pred.shape) != 4:
            raise TypeError('Model predictions shape should be (D, M, N, K), '
                            'number of datasets by number of ensemble members by number of test datapoints '
                            'by number of classes.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
                            'number of test datapoints.')
        if not issubclass(labels.dtype.type, np.integer):
            raise TypeError('Labels must be integer type.')
        K = pred.shape[3]
        if np.any(np.logical_or(labels < 0, labels >= K)):
            raise ValueError('Labels must be between 0 and K-1.')
        if np.any(np.logical_or(pred <= 1e-12, pred >= 1 - 1e-12)):
            raise ValueError('Only predictions between 0 and 1 allowed.')
        if not np.allclose(np.sum(pred, axis=-1), 1):
            pred_check = np.sum(pred, axis=-1)
            pred_check -= 1.
            pred_check = np.nan_to_num(pred_check)
            if not np.allclose(pred_check, 0):
                raise ValueError('Predictions must sum to 1 on the last axis.')

    def _generator_gradient(self, q=None):
        # The canonical link function for the Categorical.
        if q is None:
            q = self.pred
        etas = np.log(q)
        return etas - etas[:, :, :, :1]

    @staticmethod
    def _inverse_generator_gradient(etas):
        # The canonical inverse link for the Categorical.
        e = np.exp(etas)
        return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def _compute_error(pred, labels):
        return - np.log(pred[:, :, np.arange(labels.size), labels])

    def bregman_generator(self, pred):
        """
        Bregman generator for the cross-entropy loss.

        Parameters
        ----------
        pred : ndarray
            Array of vectors to which the generator is to be applied. The last dimension of the array should be
            of size corresponding to the number of classes, with the sum along that dimension being 1.

        Returns
        -------
        output : ndarray
            Array of same shape as pred.

        """
        phi = (pred * np.log(pred)).sum(axis=3)
        return phi


class SquaredLoss(BregmanDecomposition):
    """
    Create a :py:class:`BregmanDecomposition` object for squared decomposition_class regression problems with scalar-valued target.

    Parameters
    ----------
    pred : array_like
        A (D, M, N)-shape array of conditional mean estimates,
        where D is the number of training
        data resamples, M is the number of ensemble members, and
        N is the number of test data points.
    labels : array_like
        An (N,)-shape array of targets, where N is the number of test data points.

    See Also
    --------
    MultivariateSquaredLoss
    """

    def __init__(self, pred, labels):
        super().__init__(pred, labels)

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
            'number of test datapoints.')

    def _generator_gradient(self, q=None):
        if q is None:
            q = self.pred
        return 2 * q

    @staticmethod
    def _inverse_generator_gradient(etas):
        return 0.5 * etas

    @staticmethod
    def _compute_error(pred, labels):
        label_shape = np.ones_like(pred.shape)
        label_shape[-1] = -1
        labels1 = labels.reshape(label_shape)
        return (pred - labels1)**2

    def bregman_generator(self, pred):
        """
        Bregman generator for the squared loss (the square of the prediction).

        Parameters
        ----------
        pred : ndarray
            Input array.

        Returns
        -------
        output : ndarray
            Array containing `pred` squared

        """
        return pred ** 2


class MultivariateSquaredLoss(BregmanDecomposition):
    """
    Create a :py:class:`BregmanDecomposition` object given data where the target conditional
    distribution is multivariate Gaussian.

    Parameters
    ----------
    pred : array_like
        A (D, M, N, K)-shape array of conditional mean estimates,
        where D is the number of training
        data resamples, M is the number of ensemble members,
        N is the number of test data points, and K is the dimensionality
        of the target.
    labels : array_like
        An (N, K)-shape array of targets, where N is the number of test data points
        and K is the dimensionality of the target.

    See Also
    --------
    SquaredLoss
    """

    def __init__(self, pred, labels):
        super().__init__(pred, labels)

        if len(pred.shape) != 4:
            raise TypeError('Model predictions shape should be (D, M, N, K), '
            'number of datasets by number of ensemble members by number of test datapoints '
            'by dimensionality of Gaussian.')
        if len(labels.shape) != 2:
            raise TypeError('Labels shape should be (N, K), '
            'number of test datapoints by dimensionality of Gaussian.')
        if pred.shape[3] != labels.shape[1]:
            raise ValueError('Dimensionality of Gaussians do not match.')

    def _generator_gradient(self, q=None):
        if q is None:
            q = self.pred
        return 2 * q

    def bregman_generator(self, pred):
        return np.sum(pred ** 2, axis=-1)

    @staticmethod
    def _inverse_generator_gradient(etas):
        # The canonical inverse link for the isotropic MV Gaussian.
        return 0.5 * etas

    @staticmethod
    def _compute_error(pred, labels):
        label_shape = np.hstack(([1, 1], labels.shape))
        labels1 = labels.reshape(label_shape)
        return np.sum((pred - labels1)**2, axis=-1)

class PoissonLoss(BregmanDecomposition):
    """
    Create a :py:class:`Decomposition` object given that the loss function is the deviance function for the Poisson
    distribution.

    Parameters
    ----------
    pred : array_like
        A (D, M, N)-shape array of conditional mean estimates,
        where D is the number of training
        data resamples, M is the number of ensemble members, and
        N is the number of test data points.
    labels : array_like
        An (N,)-shape array of targets, where N is the number of test data points.
    """

    def __init__(self, pred, labels):
        super().__init__(pred, labels)

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
            'number of test datapoints.')
        # if not issubclass(labels.dtype.type, np.integer):
        #    raise TypeError('Labels must be integer type.')
        if np.any(labels < 0):
            raise ValueError('Labels must be non-negative.')
        if np.any(pred < 0):
            raise ValueError('Predictions must be non-negative.')

    def bregman_generator(self, pred):
        """
        The Bregman generator for the Poisson loss.

        Parameters
        ----------
        pred : ndarray
            Input array

        Returns
        -------

        """
        return pred * np.log(pred) - pred

    def _generator_gradient(self, q=None):
        if q is None:
            q = self.pred
        # The canonical link function for the Poisson.
        return np.log(q)

    @staticmethod
    def _inverse_generator_gradient(etas):
        return np.exp(etas)

    @staticmethod
    def _compute_error(pred, labels):
        #
        label_shape = np.ones_like(pred.shape)
        label_shape[-1] = -1
        labels = labels.reshape(label_shape)
        return xlogy(labels, labels / pred) - (labels - pred)

