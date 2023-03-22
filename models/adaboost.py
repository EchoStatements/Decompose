import numpy as np
import copy
from scipy.special import expit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _validate_labels_plus_minus_one(labels):
    """
    Helper function that ensures that the labels that training/scoring is performed on are (-1, 1), converting from
    (0, 1) if necessary (preserving the positive class as label 1).

    Parameters
    ----------
    labels : ndarray of shape (n_examples)
        labels whose format is to be checked

    Returns
    -------
    labels : ndarray of shape (n_examples)
        Binary labels in the set {-1,1}
    """
    # convert labels from (0, 1) to (1, 1) if necessary
    if np.isin(labels, [0, 1]).all():
        labels += -1 * (labels == 0)
    if not np.isin(labels, [-1, 1]).all():
        raise ValueError
    return labels

def _validate_labels_zero_one(labels):
    """
    Helper function that ensures that the labels that training/scoring is performed on are (0, 1), converting from
    (-1, 1) if necessary (preserving the positive class as label 1).

    labels : ndarray of shape (n_examples)
        labels whose format is to be checked

    Returns
    -------
    labels : ndarray of shape (n_examples)
        Binary labels in the set {0,1}
    """
    if not len(set(labels)) == 2:
        raise ValueError
    if np.isin(labels, [-1, 1]).all():
        labels = 1 * (labels == 1)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError
    return labels



class AdaBoost(object):
    """
    This implementation of AdaBoost was used to perform the experiments in the paper for which this library was written.
    Unless specifically wishing to measure bias, variance and diversity of the ensemble, it is recommended to use
    sklearn's AdaBoostClassifier instead

    Parameters
    ----------
    base_estimator : sklearn classifier (default=DecisionTreeClassifier(max_depth=1))
        The base estimator to be used in the AdaBoost ensemble
    n_estimators : int (default=5)
        Number of base estimators of which the ensemble is to be comprised
    normalise_estimator_weights : boolean (default=False)
        Whether the estimators weights are to be normalised to sum to one
    warm_start : boolean (default=False)
        If warm_start is True, fit adds new estimators to the ensemble up to n_estimators but keeps existing ones. If
        false, old models are overwritten and n_estimators new models are trained
    half_factor : boolean (default=True)
        There are two formulations of AdaBoost, with the weights assigned to new models differing by a factor of a half
         between the two. This parameter determines whether the factor of a half should be included or not.
    shrinkage : float (default=None)
        If this value is not None, shrinkage is applied to the ensemble members, with this value determining the size
        of shrinkage to be used
    shrinkage_first_estimator (default=False)
        If shrinkage is applied, this value determines whether the first model in the ensemble should have the shrinkage
        factor applied
    """

    def __init__(self, base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=5,
                 normalise_estimator_weights=False, warm_start=False,
                 half_factor=True,
                 shrinkage=None, shrink_first_estimator=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.normalise_estimator_weights = normalise_estimator_weights
        self.warm_start = warm_start
        self.half_factor = half_factor
        if shrinkage is not None and shrinkage > 1.:
            raise ValueError("Shrinkage should be less than 1")
        self.shrinkage = shrinkage if shrinkage is not None else 1.
        self.shrink_first_estimator = shrink_first_estimator

    def fit(self, data, labels):
        """
        Fits the AdaBoost ensemble

        Parameters
        ----------
        data - ndarray of shape (n_examples, n_features)
            Training data features
        labels - ndarray of shape (n_examples)
            Training data labels

        Returns
        -------
        None

        """
        # start with uniform weights over training data
        labels = _validate_labels_plus_minus_one(labels)
        if not hasattr(self, "sample_weight"):
            self.sample_weight = np.ones(data.shape[0]) / data.shape[0]
        if self.warm_start:
            if not hasattr(self, "data"):
                self.data = data
                self.labels = labels
            else:
                if (self.data != data).all() or (self.labels != labels).all():
                    ValueError("AdaBoost must receive same data on each iteration")
        if not hasattr(self, "estimators_"):
            self.estimators_ = []
            self.estimator_weights = []
        while len(self.estimators_) < self.n_estimators:
            new_estimator = copy.deepcopy(self.base_estimator)
            new_estimator.fit(data, labels, self.sample_weight)
            new_estimator = AdaBoostModelWrapper(new_estimator)
            estimator_error = 1. - new_estimator.score(data, labels, sample_weight=self.sample_weight)
            # new_estimator_weight = np.log((1 - estimator_error) / estimator_error)
            # Some papers use half the new_estimator weight, though this means our model no longer matches
            # sk_learn's implementation
            if self.half_factor:
                constant = 0.5
            else:
                constant = 1.
            new_estimator_weight = constant * np.log((1 - estimator_error) / estimator_error)
            if len(self.estimators_) != 0 or self.shrink_first_estimator:
                new_estimator_weight *= self.shrinkage
            self.estimator_weights.append(new_estimator_weight)
            new_estimator.weight = new_estimator_weight
            self.estimators_.append(new_estimator)
            incorrect = new_estimator.predict(data) != labels
            self.sample_weight = self.sample_weight * np.exp(new_estimator_weight * incorrect)
            # self.sample_weight = self.sample_weight * np.exp( -new_estimator.predict(data) * labels * new_estimator_weight)
            self.sample_weight /= self.sample_weight.sum()

    def decision_function(self, data):
        """
        Gives the real-valued ensemble output

        Parameters
        ----------
        data - ndarray of shape (n_examples, n_features)
            data for which ensemble output is to be generated

        Returns
        -------
        f_ens : ndarray of shape (n_examples)
            The real valued model output (i.e., the weighted sum of the individual clasifier outputs). If `normalise_estimator_weights`,
            then this value is normalised by dividing the total weight of the ensemble members.

        """
        f_ens = np.zeros(shape=data.shape[0])
        for estimator_idx in range(self.n_estimators):
            f_ens += self.estimators_[estimator_idx].predict(data) * self.estimator_weights[estimator_idx]
        if self.normalise_estimator_weights:
            f_ens = f_ens / (np.array(self.estimator_weights).sum())
        return f_ens

    def raw_model_output(self, data):
        """"
        Equivalent to `decision_function`. Deprecated.
        """
        f_ens = np.zeros(shape=data.shape[0])
        for estimator_idx in range(self.n_estimators):
            f_ens += self.estimators_[estimator_idx].predict(data) * self.estimator_weights[estimator_idx]
        if self.normalise_estimator_weights:
            f_ens = f_ens / (np.array(self.estimator_weights).sum())
        return f_ens

    def predict(self, data):
        """
        Gives the {-1, +1} prediction of the ensemble

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
            Array of examples for which predictions are to be given.

        Returns
        -------
        predictions : ndarray of shape (n_examples)
            Predicted labels for the given data.

        """
        f_ens = np.zeros(shape=data.shape[0])
        for estimator_idx in range(self.n_estimators):
            f_ens += self.estimators_[estimator_idx].predict(data) * self.estimator_weights[estimator_idx]
        # This line probably doesn't do anything anymore
        if self.normalise_estimator_weights:
            f_ens = f_ens / (np.array(self.estimator_weights).sum())
        return 1 * (f_ens >= 0) -  1 * (f_ens < 0)

    def score(self, data, labels):
        """
        The 0-1 loss of the ensemble on the given data

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
            Array of test examples

        labels : ndarray of shape (n_examples)
            Labels for test examples

        Returns
        -------
        score : float
            The 0-1 loss of the ensemble

        """
        labels = _validate_labels_plus_minus_one(labels)
        return ((self.predict(data) * labels) > 0).mean()

    def staged_predict(self, data):
        """
        Iterates over ensemble sizes, yielding the predictions for each number of estimators between 1 and n_estimators (inclusive).

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
            Array of examples

        Returns
        -------

        """
        f_ens = np.zeros(data.shape[0])
        for estimator in self.estimators_:
            # TODO: here we are assuming that we use the half factor, it should be optional
            f_ens += 0.5 * estimator.decision_function(data)
            yield np.sign(f_ens)


class AdaBoostModelWrapper(object):
    """
    A wrapper for the base_estimator used in AdaBoost, standardises predict, decision_function etc for use with
    BVDExperiment class
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model
        """
        self.model = model
        self.weight = None

    def predict(self, data):
        """
        The predicted class of the examples given in `data`

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)

        Returns
        -------
        predictions : ndarray of shape (n_examples)
            Predictions for examples in data

        """
        return self.model.predict(data)

    def decision_function(self, data):
        """
        The predicted class multiplied by the weight of model in the AdaBoost ensemble

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)

        Returns
        -------
        model_outputs : ndarray of shape (n_examples)
            The {-1, 1} predictions of the base_estimator multiplied by the model weight

        """
        return self.model.predict(data) * self.weight

    def score(self, data, labels, **kwargs):
        """
        The classification accuracy of the model on dataset (data, labels).

        Parameters
        ----------
        data - ndarray of shape (n_examples, n_features)
        labels - ndarray of shape (n_examples, )

        Returns
        -------
        score : float
            0-1 loss of the base model on the given data and ground truth labels.

        """
        return self.model.score(data, labels, **kwargs)

    def get_params(self):
        """
        Used to get parameters inside the model.

        Returns
        -------

        """
        return self.model.get_params()

    def set_params(self, **params):
        """
        Used to set parameters inside the model.

        Parameters
        ----------
        params

        Returns
        -------

        """
        self.model.set_params(**params)
        return self

class LogitBoost(object):
    """
    Implementation of LogitBoost based on https://github.com/artemmavrin/logitboost
    The original implementation is to be preferred over this one unless specifically for use with the `BVDExperiment`
    class.

    Parameters
    ----------
    base_estimator : sklearn regressor (default=DecisionTreeRegressor(max_depth=1))
        The base estimator to be used in the LogitBoost ensemble
    n_estimators : int (default=5)
        Number of base estimators of which the ensemble is to be comprised
    warm_start : boolean (default=False)
        If warm_start is True, fit adds new estimators to the ensemble up to n_estimators but keeps existing ones. If
        false, old models are overwritten and n_estimators new models are trained
    max_response_magnitude : float (default=4.)
        Clips the maximum response value of ensemble members for the purposes of stability
    shrinkage : float (default=None)
        If this value is not None, shrinkage is applied to the ensemble members, with this value determining the size
        of shrinkage to be used
    shrinkage_first_estimator (default=False)
        If shrinkage is applied, this value determines whether the first model in the ensemble should have the shrinkage
        factor applied
    weight_trim_quantile : float (default=0.05)
        Weights below this value are set to zero
    """

    def __init__(self, base_estimator=DecisionTreeRegressor(max_leaf_nodes=2),
                 n_estimators=5, warm_start=False, max_response_magnitude=4,
                 shrinkage=None, shrink_first_estimator=False, weight_trim_quantile=0.05):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.warm_start = warm_start
        self.max_response_magnitude = max_response_magnitude
        self.weight_trim_quantile = weight_trim_quantile
        if shrinkage is not None and shrinkage > 1.:
            raise ValueError("Shrinkage should be less than 1")
        self.shrinkage = shrinkage if shrinkage is not None else 1.
        self.shrink_first_estimator = shrink_first_estimator

    def fit(self, data, labels):
        """
        Fits the LogitBoost ensemble to the training data (data, labels)

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
        labels : ndarray of shape (n_examples)

        Returns
        -------
        None

        """
        if self.warm_start:
            if not hasattr(self, "data"):
                self.data = data
                self.labels = labels
            else:
                if (self.data != data).all() or (self.labels != labels).all():
                    ValueError("LogitBoost must receive same data on each iteration")
        labels = _validate_labels_zero_one(labels)
        if not hasattr(self, "estimators_"):
            self.sample_weight = np.ones(data.shape[0]) / data.shape[0]
            self.sample_probabilities = np.ones(data.shape[0]) * 0.5
            self.estimators_ = []
            self.sample_output = np.zeros(data.shape[0])
        while len(self.estimators_) < self.n_estimators:
            self.sample_weight = (self.sample_probabilities) * (1. - self.sample_probabilities)
            self.sample_weight = np.maximum(self.sample_weight,  2. * np.finfo(float).eps)
            with np.errstate(divide='ignore', over='ignore'):
                response = np.where(labels, 1 / self.sample_probabilities, -1. / (1. - self.sample_probabilities))
            response = np.clip(response,
                               a_min=-self.max_response_magnitude,
                               a_max=self.max_response_magnitude)
            new_estimator = copy.deepcopy(self.base_estimator)
            if self.weight_trim_quantile is not None:
                threshold = np.quantile(self.sample_weight, self.weight_trim_quantile, interpolation="lower")
                mask = (self.sample_weight >= threshold)
                new_estimator.fit(data[mask], response[mask], sample_weight=self.sample_weight[mask])
            else:
                new_estimator.fit(data, response, sample_weight=self.sample_weight)

            if len(self.estimators_) != 0 or self.shrink_first_estimator:
                new_estimator = LogitBoostModelWrapper(new_estimator, shrinkage=self.shrinkage)
            else:
                new_estimator = LogitBoostModelWrapper(new_estimator, shrinkage=1.)
            self.sample_output += new_estimator.decision_function(data)
            self.sample_probabilities = expit(self.sample_output)
            self.estimators_.append(new_estimator)
            if (self.sample_probabilities == 0.).any():
                print(np.max(self.sample_output * 2))
                print("FOUND UNDERFLOW")

    def predict(self, data):
        """
        Predict the classes of the examples given in `data`

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)

        Returns
        -------
        predictions : ndarray of shape (n_examples)
            Predictions for given data

        """
        f_ens = np.zeros(data.shape[0])
        for estimator in self.estimators_:
            f_ens += estimator.decision_function(data)
        return 1 * (f_ens >= 0) -  1 * (f_ens < 0)

    def raw_model_output(self, data):
        """

        Gives the real-valued ensemble output. Deprecated (decision_function should be used instead)

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)

        Returns
        -------
        f_ens : ndarray of shape (n_examples)
            Real valued output of the ensemble.

        """
        f_ens = np.zeros(shape=data.shape[0])
        for estimator_idx in range(self.n_estimators):
            f_ens += self.estimators_[estimator_idx].decision_function(data)
        return f_ens

    def decision_function(self, data):
        """
        Gives the real-valued ensemble output

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)

        Returns
        -------
        f_ens : ndarray of shape (n_examples)
            Real valued output of the ensemble.
        """
        f_ens = np.zeros(data.shape[0])
        for estimator in self.estimators_:
            f_ens += estimator.decision_function(data)
        return f_ens

    def staged_decision_function(self, data):
        """
        Iteratively yields decision functions for ensemble sizes of 1 to n_estimators (inclusive).

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
            Data for which we want model output

        Returns
        -------
        f_ens : ndarray of shape (n_examples)
            Real valued output of the ensemble


        """
        f_ens = np.zeros(data.shape[0])
        for estimator in self.estimators_:
            f_ens += estimator.decision_function(data)
            yield f_ens

    def staged_predict(self, data):
        """
        Iteratively yields predictions for ensemble sizes of 1 to n_estimators (inclusive).

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
            Data for which we want model output

        Returns
        -------
        predictions : ndarray of shape (n_examples)
            Predicted class labels for data


        """
        f_ens = np.zeros(data.shape[0])
        for estimator in self.estimators_:
            f_ens += estimator.decision_function(data)
            yield np.sign(f_ens)


    def part_score(self, data, labels, num_estimators):
        """
        The 0-1 loss of the sub-ensemble of size num_estimators on the dataset (data, labels)

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
        labels : ndarray of shape (n_examples,)
        num_estimators

        Returns
        -------
        score : float
            The score of the ensemble of size n_estimators on the given data

        """
        labels = _validate_labels_plus_minus_one(labels)
        f_ens = np.zeros(data.shape[0])
        for est_idx in range(num_estimators):
            f_ens += self.estimators_[est_idx].decision_function(data)
        return ((f_ens * labels) > 0).mean()

    def score(self, data, labels):
        """

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
        labels : ndarray of shape (n_exampes,)

        Returns
        -------
        score : float
            The 0-1 loss of the ensemble on the given data

        """
        labels = _validate_labels_plus_minus_one(labels)
        if not np.isin(labels, [-1, 1]).all():
            raise ValueError
        return ((self.decision_function(data) * labels) > 0).mean()

    def staged_score(self, data, labels):
        """
        Iteratively gives the 0-1 loss for ensembles of sizes 1 to n_estimators (inclusive)

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
        labels : ndarray of shape (n_exampes,)

        Returns
        -------
        score : float
            The 0-1 loss of the ensemble on the given data

        """
        labels = _validate_labels_plus_minus_one(labels)
        for pred in self.staged_decision_function(data):
            yield ((pred * labels) > 0).mean()

class LogitBoostModelWrapper(object):
    """
    A wrapper for the base_estimator used in AdaBoost, standardises predict, decision_function etc for use with
    BVDExperiment class
    """

    def __init__(self, model, shrinkage=None):
        self.model = model
        self.shrinkage = 1. if shrinkage is None else shrinkage

    def predict(self, data):
        """
        The class prediction (not the regression prediction) of the base regression model

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)
            Array of examples whose classes are to be predicted

        Returns
        -------

        """
        return np.sign(self.model.predict(data))

    def decision_function(self, data):
        """

        Parameters
        ----------
        data : ndarray of shape (n_examples, n_features)

        Returns
        -------

        """
        return self.shrinkage * self.model.predict(data)

    def score(self, data, labels, **kwargs):
        return self.model.score(data, labels, **kwargs)

    def get_params(self):
        """
        Used to get parameters inside the model.

        Returns
        -------

        """
        return self.model.get_params()

    def set_params(self, **params):
        """
        Used to set parameters inside the model.

        Parameters
        ----------
        params : dict
            Dictionary of parameters to be set

        Returns
        -------

        """
        self.model.set_params(**params)
        return self
