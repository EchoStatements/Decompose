"""
This file contains helper classes to do regression testing for the switch from exponential family decomposition_class
to Bregman decomposition_class
"""


from cached_property import cached_property
import numpy as np
from functools import partial
from scipy.stats import mode
from scipy.special import factorial


class Decomposition(object):
    """
    A base class for decomposition_class. See the subclasses in the
    :ref:`Distributions module<distributions_module>`
    for a description of the inputs.

    See Also
    --------
    Gaussian, MultivariateGaussian, Bernoulli, Categorical,
    Poisson
    """
    def __init__(self, pred, labels):
        if pred.shape[2] != labels.shape[0]:
            raise ValueError('Third dimension of model predictions should match '
                             'first dimension of labels (# test data points)')

        self.pred = pred
        self.labels = labels

        # Compute the natural parameters
        self.etas = self._canonical_link()

        self.num_training_data_resamples = pred.shape[0]
        self.num_ensemble_members = pred.shape[1]
        self.num_test_examples = pred.shape[2]

    def _NLL(self, axes1=(), axes2=()):
        # For each of the expected error/bias terms in the double decomposition_class we:
        # 1) Average the outputs over one or more axes;
        # 2) Apply the inverse link function, and compute a likelihood;
        # 3) Average over zero or more axes.
        #
        # This function generalizes this, allowing it to be used to compute each
        # of the expected error and bias terms.

        x = self._canonical_inverse_link(
            self.etas.mean(axis=axes1, keepdims=True)
        )

        NLL = self._compute_NLL(x, self.labels)
        return np.mean(NLL, axis=axes2).squeeze()

    def _KL_difference(self, axes1=(), axes2=(), axes3=()):
        # For each of the variance-like terms in the double decomposition_class we:
        # 1) Average the outputs over zero or more axes;
        # 2) Compute the Jensen gap of the log-normalizer function,
        #    E[A(X)] - A(E[X]), where the expectation is over one or more axes;
        # 3) Average over the remaining axes, apart from the test data points axis.
        #
        # This function generalizes this, allowing it to be used to compute each
        # of the the variance-like terms.

        x = self.etas.mean(axis=axes1, keepdims=True)
        E = partial(np.mean, axis=axes2, keepdims=True)
        A = self._A

        diff = E(A(x)) - A(E(x))
        return diff.mean(axis=axes3).squeeze()

    @cached_property
    def expected_ensemble_risk(self):
        """
        Compute the expected ensemble risk (with respect to training datasets)
        for each test data point, :math:`\\mathbb{E}_D[ NLL (\\bar{q})]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._NLL(1, 0)

    @cached_property
    def ensemble_central_pred_risk(self):
        """
        Compute the risk of the central ensemble for each test data point
        This is equal to the bias of the central ensemble plus the conditional entropy H(Y|X)
        :math:`NLL(\\bar{q}^*)`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._NLL((0, 1))

    @cached_property
    def ensemble_variance(self):
        """
        Compute the ensemble variance for each test data point,
        :math:`\\mathbb{E}_D[K(\\bar{q}^*, \\bar{q})]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._KL_difference(1, 0)

    @cached_property
    def expected_average_model_risk(self):
        """
        Compute the expected average model risk for each test data point,
        :math:`\\mathbb{E}_D[ \\frac{1}{M} \\sum_{i=1}^M NLL(q_i)]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._NLL(axes2=(0, 1))

    @cached_property
    def average_central_risk(self):
        """
        Compute the average risk of the central models for each test data point,
        This is equal to the average bias of the central models plus the conditional entropy H(Y|X)
        :math:`\\frac{1}{M} \\sum_{i=1}^M NLL(q_i^*)`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._NLL(0, 1)

    @cached_property
    def average_model_variance(self):
        """
        Compute the average model variance for each test data point,
        :math:`\\frac{1}{M} \\sum_{i=1}^M \\mathbb{E}_D[K(q_i^*, q_i)]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._KL_difference((), 0, 1)

    @cached_property
    def disparity(self):
        """
        Compute the disparity for each test data point,
        :math:`\\frac{1}{M} \\sum_{i=1}^M K(\\bar{q}^*, q_i^*)`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._KL_difference(0, 1)

    @cached_property
    def diversity(self):
        """
        Compute the diversity for each test data point,
        :math:`\\mathbb{E}[\\frac{1}{M} \\sum_{i=1}^M K(\\bar{q}, q_i)]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._KL_difference((), 1, 0)



class Gaussian(Decomposition):
    """
    Create a :py:class:`Decomposition` object given data where the target conditional
    distribution is univariate Gaussian.

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
    MultivariateGaussian
    """

    def __init__(self, pred, labels):
        super().__init__(pred, labels)

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
            'number of test datapoints.')

    def _canonical_link(self):
        # The canonical link function for the fixed-variance Gaussian.
        return self.pred

    @staticmethod
    def _canonical_inverse_link(etas):
        # The canonical inverse link for the fixed-variance Gaussian.
        return etas

    @staticmethod
    def _compute_NLL(pred, labels):
        label_shape = np.ones_like(pred.shape)
        label_shape[-1] = -1
        labels1 = labels.reshape(label_shape)
        # The negative log-likelihood for the unit-variance Gaussian,
        return (pred - labels1)**2 / 2 #+ np.log(2 * np.pi) / 2

    @staticmethod
    def _A(etas):
        # The log-normalizer for the fixed-variance Gaussian,
        return etas**2 / 2 # + np.log(2 * np.pi) / 2


class MultivariateGaussian(Decomposition):
    """
    Create a :py:class:`Decomposition` object given data where the target conditional
    distribution is multivariate Gaussian.

    Parameters
    ----------
    pred : array_like
        A (D, M, N, K)-shape array of conditional mean estimates,
        where D is the number of training
        data resamples, M is the number of ensemble members,
        N is the number of test data points, and K is the dimensionality
        of the Gaussian.
    labels : array_like
        An (N, K)-shape array of targets, where N is the number of test data points
        and K is the dimensionality of the Gaussian.

    See Also
    --------
    Gaussian
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

    def _canonical_link(self):
        # The canonical link function for the isotropic MV Gaussian.
        return self.pred

    @staticmethod
    def _canonical_inverse_link(etas):
        # The canonical inverse link for the isotropic MV Gaussian.
        return etas

    @staticmethod
    def _compute_NLL(pred, labels):
        # The negative log-likelihood for the isotropic MV Gaussian
        label_shape = np.hstack(([1, 1], labels.shape))
        labels1 = labels.reshape(label_shape)
        return np.sum((pred - labels1)**2, axis=-1) / 2

    @staticmethod
    def _A(etas):
        # The isotropic MV Gaussian log-normalizer,
        return np.sum(etas**2, axis=-1) / 2


class Bernoulli(Decomposition):
    """
    Create a :py:class:`Decomposition` object given data where the target conditional
    distribution is Bernoulli.

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
    Categorical
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

    def _canonical_link(self):
        # The canonical link function for the Bernoulli.
        return np.log(self.pred) - np.log(1 - self.pred)

    @staticmethod
    def _canonical_inverse_link(etas):
        # The canonical inverse link for the Bernoulli.
        return 1 / (1 + np.exp(-etas))

    @staticmethod
    def _compute_NLL(pred, labels):
        label_shape = np.ones_like(pred.shape)
        label_shape[-1] = -1
        labels1 = labels.reshape(label_shape)
        # The negative log-likelihood for the Bernoulli.
        return - labels1 * np.log(pred) - (1 - labels1) * np.log(1 - pred)

    @staticmethod
    def _A(etas):
        # The log-normalizer for the Bernoulli.
        return np.log(1 + np.exp(etas))

    @cached_property
    def average_model_classification_error_rate(self):
        """
        Compute the average error rate of the classifier ensemble members,
        i.e., the zero-one decomposition_class averaged over ensemble members and test
        data points.

        Returns
        ----------
        array_like
            A (D,)-shape array with one value per training dataset resample.
        """
        # The predictions for each ensemble member.
        # Shape: (Datasets, Ensemble members, Test data points)
        predictions = (self.etas > 0).astype(np.int32)

        # Reshape the labels so broadcasting works
        label_shape = np.ones_like(self.etas.shape)
        label_shape[-1] = -1
        labels = self.labels.reshape(label_shape)

        # Where are the predictions not equal to the labels
        error_rates = predictions != labels
        # Average over ensemble members and test data points
        return np.mean(error_rates, axis=(1, 2))

    @cached_property
    def product_ensemble_classification_error_rate(self):
        """
        Compute the error rate of the product classifier ensemble,
        i.e., the zero-one ensemble decomposition_class averaged over test
        data points.

        Returns
        ----------
        array_like
            A (D,)-shape array with one value per training dataset resample.
        """
        # Ensemble natural parameters
        eta_bar = np.mean(self.etas, axis=1)
        # The predictions of the ensemble.
        # Shape: (Datasets, Test data points)
        predictions = (eta_bar > 0).astype(np.int32)

        # Reshape the labels so broadcasting works
        label_shape = np.ones_like(eta_bar.shape)
        label_shape[-1] = -1
        labels = self.labels.reshape(label_shape)

        # Where are the predictions not equal to the labels
        error_rates = predictions != labels
        # Average over test data points
        return np.mean(error_rates, axis=1)

    @cached_property
    def voting_ensemble_classification_error_rate(self):
        """
        Compute the error rate of the voting classifier ensemble,
        i.e., the zero-one ensemble decomposition_class averaged over test
        data points.

        Returns
        ----------
        array_like
            A (D,)-shape array with one value per training dataset resample.
        """

        # Ensemble member predictions
        member_predictions = (self.etas > 0).astype(np.int32)
        # Voting prediction, shape (D, N)
        predictions = mode(member_predictions, axis=1)[0].squeeze()

        # Reshape the labels so broadcasting works
        label_shape = np.ones_like(predictions.shape)
        label_shape[-1] = -1
        labels = self.labels.reshape(label_shape)

        # Where are the predictions not equal to the labels
        error_rates = predictions != labels

        # Average over test data points
        return np.mean(error_rates, axis=1)


class Categorical(Decomposition):
    """
    Create a :py:class:`Decomposition` object given data where the target conditional
    distribution is Categorical.

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
    Bernoulli
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
            # Hacky thing to not throw an error when there are nans in the prediction
            pred_check = np.sum(pred, axis=-1)
            pred_check -= 1.
            pred_check = np.nan_to_num(pred_check)
            if not np.allclose(pred_check, 0):
                raise ValueError('Predictions must sum to 1 on the last axis.')

    def _canonical_link(self):
        # The canonical link function for the Categorical.
        etas = np.log(self.pred)
        return etas - etas[:, :, :, :1]

    @staticmethod
    def _canonical_inverse_link(etas):
        # The canonical inverse link for the Categorical.
        e = np.exp(etas)
        return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def _compute_NLL(pred, labels):
        # The negative log-likelihood for the Categorical.
        return - np.log(pred[:, :, np.arange(labels.size), labels])

    @staticmethod
    def _A(etas):
        # The log-normalizer for the Categorical.
        return np.log(np.sum(np.exp(etas), axis=-1))

    @cached_property
    def average_model_classification_error_rate(self):
        """
        Compute the average error rate of the classifier ensemble members,
        i.e., the zero-one decomposition_class averaged over ensemble members and test
        data points.

        Returns
        ----------
        array_like
            A (D,)-shape array with one value per training dataset resample.
        """
        # The predictions for each ensemble member.
        # Shape: (Datasets, Ensemble members, Test data points)
        predictions = np.argmax(self.etas, axis=-1)

        # Reshape the labels so broadcasting works
        label_shape = np.ones_like(predictions.shape)
        label_shape[-1] = -1
        labels = self.labels.reshape(label_shape)

        # Where are the predictions not equal to the labels
        error_rates = predictions != labels
        # Average over ensemble members and test data points
        return np.mean(error_rates, axis=(1, 2))

    @cached_property
    def product_ensemble_classification_error_rate(self):
        """
        Compute the error rate of the product classifier ensemble,
        i.e., the zero-one ensemble decomposition_class averaged over test
        data points.

        Returns
        ----------
        array_like
            A (D,)-shape array with one value per training dataset resample.
        """
        # Ensemble natural parameters
        eta_bar = np.mean(self.etas, axis=1)
        # The predictions for the ensemble.
        # Shape: (Datasets, Test data points)
        predictions = np.argmax(eta_bar, axis=-1)

        # Reshape the labels so broadcasting works
        label_shape = np.ones_like(predictions.shape)
        label_shape[-1] = -1
        labels = self.labels.reshape(label_shape)

        # Where are the predictions not equal to the labels
        error_rates = predictions != labels
        # Average over test data points
        return np.mean(error_rates, axis=1)

    @cached_property
    def voting_ensemble_classification_error_rate(self):
        """
        Compute the error rate of the voting classifier ensemble,
        i.e., the zero-one ensemble decomposition_class averaged over test
        data points.

        Returns
        ----------
        array_like
            A (D,)-shape array with one value per training dataset resample.
        """
        # Individual ensemble member predictions
        member_predictions = np.argmax(self.etas, axis=-1)
        # Voting prediction, shape (D, N)
        predictions = mode(member_predictions, axis=1)[0].squeeze()

        # Reshape the labels so broadcasting works
        label_shape = np.ones_like(predictions.shape)
        label_shape[-1] = -1
        labels = self.labels.reshape(label_shape)

        # Where are the predictions not equal to the labels
        error_rates = predictions != labels
        # Average over test data points
        return np.mean(error_rates, axis=1)


class Poisson(Decomposition):
    """
    Create a :py:class:`Decomposition` object given data where the target conditional
    distribution is Poisson.

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
        if not issubclass(labels.dtype.type, np.integer):
            raise TypeError('Labels must be integer type.')
        if np.any(labels < 0):
            raise ValueError('Labels must be non-negative.')
        if np.any(pred < 0):
            raise ValueError('Predictions must be non-negative.')

    def _canonical_link(self):
        # The canonical link function for the Poisson.
        return np.log(self.pred)

    @staticmethod
    def _canonical_inverse_link(etas):
        # The canonical inverse link for the Poisson.
        return np.exp(etas)

    @staticmethod
    def _compute_NLL(pred, labels, use_stirling=False):
        #TODO: tests for this
        #
        # The negative log-likelihood for the Poisson,
        # use_stirling=True uses the Stirling approximation to the factorial term in the NLL.
        label_shape = np.ones_like(pred.shape)
        label_shape[-1] = -1
        labels1 = labels.reshape(label_shape)
        if use_stirling:
            stirling = labels[labels > 0]
            stirling = stirling * np.log(stirling) - stirling
            stirling1 = np.zeros_like(labels)
            stirling1[labels > 0] = stirling
            stirling1 = stirling1.reshape(label_shape)
            nll = - labels1 * np.log(pred) + pred + stirling1
        else:
            nll = - labels1 * np.log(pred) + pred + np.log(factorial(labels1, exact=True))
        return nll

    @staticmethod
    def _A(etas):
        # The log-normalizer for the Poisson
        return np.exp(etas)



def unweighted_mode(x, axis=-1):
    """
    This is used for regression test to make sure that the weighted mode acts correctly in the unweighted case
    Parameters
    ----------
    x
    axis

    Returns
    -------

    """
    # x is expected to be some ndarray
    if not x.dtype == int:
        print("Not int")

    num_classes = np.max(x) + 1

    counts = np.apply_along_axis(np.bincount, axis, x, minlength=num_classes)

    maxes_mask = (counts == np.amax(counts, axis=axis, keepdims=True))
    argmaxes = np.argmax(maxes_mask, axis=axis)
    argmaxes = np.expand_dims(argmaxes, axis)
    return argmaxes
