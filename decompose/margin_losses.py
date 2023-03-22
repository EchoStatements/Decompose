import numpy as np

from decompose import MarginDecomposition, BregmanDecomposition

class LogisticMarginLoss(MarginDecomposition):

    def __init__(self, pred, labels, is_additive=True):
        self.pred = pred
        self.labels = labels
        self.is_additive = is_additive
        self.is_margin_decomposition = True

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
                            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
                            'number of test datapoints.')

        if pred.shape[2] != labels.shape[0]:
            raise ValueError('Third dimension of model predictions should match '
                             'first dimension of labels (# test data points)')
        self.num_training_data_resamples = pred.shape[0]
        self.num_ensemble_members = pred.shape[1]
        self.num_test_examples = pred.shape[2]
        if self.is_additive:
            pred *= self.num_ensemble_members

    @staticmethod
    def bregman_generator(pred):
        pred_pos = (pred > 0) * pred
        pred_neg = (pred <= 0) * pred
        out = (pred > 0 ) * np.log(1. +  np.exp(-pred_pos)) + (pred <= 0) * (-pred + np.log(1. + np.exp(pred_neg)))
        # assert np.allclose(out, np.log(1 + np.exp(-pred)))
        return out

    @staticmethod
    def _compute_error(pred, labels):
        f_margin = pred * labels
        f_pos = (f_margin > 0) * f_margin
        f_neg = (f_margin <= 0) * f_margin
        out = (f_margin > 0 ) * np.log(1. +  np.exp(-f_pos)) + (f_margin <= 0) * (-f_margin + np.log(1. + np.exp(f_neg)))
        # assert np.allclose(out, np.log(1 + np.exp(-pred * labels)))
        return out

    def _generator_gradient(self, f=None):
        f_pos = (f > 0) * f
        f_neg = (f <= 0) * f
        if f is None:
            f = self.pred
        out = -1 * ((f > 0) * np.exp(-f_pos) / (1 + np.exp(-f_pos)) + (f <= 0) * 1 / (1 + np.exp(f_neg)))
        # assert np.allclose(out, - 1. / (np.exp(f) + 1))
        return out

class BujaLogisticMarginLoss(BregmanDecomposition):

    def __init__(self, pred, labels):
        raise NotImplementedError

        self.pred_f = pred
        self.pred = self._inverse_optimal_link(pred)
        self.etas = self._generator_gradient()
        self.labels = labels

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
                            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
                            'number of test datapoints.')
        # TODO: Check that labels are in the correct form
        if pred.shape[2] != labels.shape[0]:
            raise ValueError('Third dimension of model predictions should match '
                             'first dimension of labels (# test data points)')

        self.n_training_data_resamples = pred.shape[0]
        self.n_ensemble_members = pred.shape[1]
        self.n_test_examples = pred.shape[2]

    @staticmethod
    def bregman_generator(pred):
        return pred * np.log(pred) + (1. - pred) * np.log(1. - pred)

    @staticmethod
    def _compute_error(pred, labels):
        pred = np.log(pred/ (1. - pred))
        return np.log(1. + np.exp(-pred * labels))
        #return  -((labels == 1) * np.log(pred) + (labels == -1) * np.log(1 - pred))

    def _generator_gradient(self, q=None):
        if q is None:
            q = self.pred
        return np.log(q / (1. - q))

    def _inverse_generator_gradient(self, etas):
        return np.exp(etas) / (1. + np.exp(etas))

    def _optimal_link(self, f):
        return self._generator_gradient(self, f)

    def _inverse_optimal_link(self, q):
        return  self._inverse_generator_gradient(q)

    def _expected_risk(self):
        return ((self.labels == 1) * np.log(1 + np.exp(-self.pred))
                 + (self.labels == -1) * np.log(1 + np.exp(self.pred))).mean(axis=0).squeeze()


class ExponentialMarginLoss(MarginDecomposition):

    def __init__(self, pred, labels, is_additive=True):
        self.pred = pred
        self.labels = labels
        self.is_margin_decomposition = True
        self.is_additive = is_additive

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
                            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
                            'number of test datapoints.')

        if pred.shape[2] != labels.shape[0]:
            raise ValueError('Third dimension of model predictions should match '
                             'first dimension of labels (# test data points)')
        self.num_training_data_resamples = pred.shape[0]
        self.num_ensemble_members = pred.shape[1]
        self.num_test_examples = pred.shape[2]

        if self.is_additive:
            self.pred *= self.num_ensemble_members

    @staticmethod
    def bregman_generator(pred):
        return np.exp(-pred)

    @staticmethod
    def _compute_error(pred, labels):
        return np.exp(-pred * labels)

    def _generator_gradient(self, f=None):
        if f is None:
            f = self.pred
        return -np.exp(-f)


class BujaExponentialLoss(BregmanDecomposition):
    """
    Create a :py:class:`BregmanDecomposition` for margin classifier with exponential decomposition_class

    Parameters
    ----------
    pred : array_like
        A (D, M, N)-shape array of conditional mean estimates,
        where D is the number of training
        data resamples, M is the number of ensemble members and
        N is the number of test data points.
    labels : array_like

    See Also
    --------
    BinaryCrossEntropy

    """
    def __init__(self, pred, labels):

        self.raw_pred = pred
        self.pred = self._inverse_optimal_link(pred)
        self.etas = self._generator_gradient()
        self.labels = labels

        if len(pred.shape) != 3:
            raise TypeError('Model predictions shape should be (D, M, N), '
                            'number of datasets by number of ensemble members by number of test datapoints.')
        if len(labels.shape) != 1:
            raise TypeError('Labels shape should be (N,), '
                            'number of test datapoints.')
        # TODO: Check that labels are in the correct form
        if pred.shape[2] != labels.shape[0]:
            raise ValueError('Third dimension of model predictions should match '
                             'first dimension of labels (# test data points)')

        self.num_training_data_resamples = pred.shape[0]
        self.num_ensemble_members = pred.shape[1]
        self.num_test_examples = pred.shape[2]

    @staticmethod
    def bregman_generator(pred):
        return -2 * np.sqrt(pred * (1. - pred))

    @staticmethod
    def loss_function(v):
        return np.exp(-v)

    @staticmethod
    def _loss_gradient(v):
        return - np.exp(-v)

    @staticmethod
    def _compute_error(pred, labels):
        # TODO: lots of testing and compatibility stuff needs to go here.
        return np.sqrt((1. - pred) / pred) ** labels
        # return 2 * np.sqrt(pred * (1. - pred)) + ((2. * pred - 1.) / np.sqrt(pred * (1 - pred))) * (pred - (labels > 0))

    def _expected_risk(self):
        # x = self._optimal_link(self.pred)
        # return ((self.labels == 1) * np.exp(-x) + (self.labels == -1) * np.exp(x)).mean(axis=0).squeeze()
        return ((self.labels == 1) * np.exp(-self.raw_pred) + (self.labels == -1) * np.exp(self.raw_pred)).mean(axis=0).squeeze()

    def _inverse_optimal_link(self, q):
        return 1. / (1 + np.exp(-2 * q))

    def _optimal_link(self, f):
        return 0.5 * np.log(f / (1-f))

    def _generator_gradient(self, f=None):
        if f is None:
            f = self.pred
        return (2. * f - 1.) / np.sqrt(f * (1 - f))

    @staticmethod
    def _inverse_generator_gradient(etas):
        # return ((4 + etas ** 2) + etas * np.sqrt(4 + etas ** 2)) / (8 + 2 * (etas ** 2))
        return 0.5 + etas / (2 * np.sqrt(etas ** 2 + 4))
