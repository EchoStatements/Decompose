from decompose import EffectDecomposition
import numpy as np
from numpy.random import default_rng


def mode_with_random_ties(x, axis=-1, weights=None, random_ties=True, random_seed="axis"):
    """
    Computes the mode of an array over the given axis. Function allows for different weightings for different models
    and also allows for two tie break procedures: ties are either broken at random (when `random_ties=True`) or
    using np.argmax's default behaviour: choosing the lowest class label in the set of tied classes.


    Parameters
    ----------
    x : integer ndarray
        The data for which the mode is to be found
    axis : int (default=-1)
        The axis along which the mode is to be calculated. By default, this is the last axis of the array
    weights : ndarray of same shape as x
        The relative weights of each entry of x
    random_ties : boolean (default=True)
        If True, when there are two or more modal values, the one to be returned is chosen at random from those values.
        If False, the default behaviour of np.argmax is used (by default this is the lowest value).
    random_seed : int or "axis" or None
        The random seed used for the RNG in the tie-breaker. If set to the string "axis", the integer value of the axis
        argument is used. This allows for the decomposition object to have statistically independent tie breaks in different
        axes in it's predictions by setting different seeds for breaking ties over the :math:`M` ensemble members and the :math:`D` trials.

    Returns
    -------
    argmaxes : ndarray
        An array of dimension x.ndim-1 giving the modal values along the axis `axis`.
    """

    # Check first that we are dealing with ints
    if not x.dtype == int:
        print("Datatype of x is not int, this may give unexpected behaviour")

    num_classes = int(np.max(x) + 1)

    # We get a matrix of the same size as the original, where each
    if weights is None:
        weights = np.ones_like(x).astype(int)
    new_shape = list(x.shape)
    new_shape[axis] = 0
    counts = np.zeros(new_shape)
    # Get masked weights for each class
    for class_label in range(num_classes):
        slice = ((x == class_label) * weights).sum(axis=axis, keepdims=True)
        counts = np.concatenate((counts, slice), axis=axis)


    maxes_mask = (counts == np.amax(counts, axis=axis, keepdims=True))
    if random_ties:
        if random_seed == "axis":
            random_seed = axis
        rng = default_rng(random_seed)
        argmaxes = np.argmax(maxes_mask * rng.random(size=maxes_mask.shape), axis=axis)
    else:
        argmaxes = np.argmax(maxes_mask, axis=axis)
    argmaxes = np.expand_dims(argmaxes, axis)
    return argmaxes


class ZeroOneLoss(EffectDecomposition):
    """
    Create a :py:class:`EffectDecomposition` object using the 0-1 loss

    Parameters
    ----------
    pred : array_like
        A (D, M, N)-shape array of class predictions,
        where D is the number of training
        data resamples, M is the number of ensemble members, and
        N is the number of test data points.
    labels : array_like
        An (N,)-shape array of integer labels (0, 1, ..., k), where N is the number of
        test data points.

    See Also
    --------
    CrossEntropy
    """
    def __init__(self, pred, labels, weights=None, random_ties=False):
        self.pred = pred
        self.labels = labels
        self.weights = weights
        self.random_ties = random_ties
        # For 0-1, models are combined using the majority vote

    def aggregator(self, preds, axis=(), weights=None):
        """

        Parameters
        ----------
        preds : ndarray
        axis : axes along which to perform aggregation
            Axes along which aggregation is to be performed. Note that unlike with the centroid combiner of Bregman
            divergences, the order in which axes are aggregated across matters.
        weights : None ndarray of same shape as preds
            Relative weighting to be given to each prediction

        Returns
        -------

        """
        if isinstance(axis, int):
            axis = (axis,)
        result = preds
        for ax in axis:
            result = mode_with_random_ties(result, axis=ax, weights=weights, random_ties=self.random_ties)
        return result

    @staticmethod
    def _compute_error(pred, labels):
        """

        Parameters
        ----------
        pred : ndarray
            array of predictions where first dimension is of size n_examples
        labels : ndarray
            array ground truth labels of shape (n_examples,)

        Returns
        -------
        error : ndarray
           array of same size as pred, with entries as 1. when prediction and ground truth agree and 0. otherwise.

        """
        return 1. * (pred != labels)

