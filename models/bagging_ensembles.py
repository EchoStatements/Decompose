import copy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble._bagging import  BaseBagging, BaggingClassifier
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.fixes import delayed
from sklearn.ensemble._base import _set_random_states
from sklearn.base import clone
from sklearn.utils import check_random_state

from decompose.experiments import _add_model_smoothing

from joblib import Parallel

import numpy as np
from sklearn.utils.validation import check_is_fitted


def _parallel_predict_log_proba_geometric(estimators, estimators_features, X, n_classes,
                                          probability_epsilon):
    """Private function used to compute log probabilities within a job.
        Like _parallel_predict_log_proba, except computes the sum of the logs, rather than using logaddexp
    """
    n_samples = X.shape[0]
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(0)

    for estimator, features in zip(estimators, estimators_features):
        proba_estimator = estimator.predict_proba(X[:, features])

        if probability_epsilon is not None:
            proba_estimator = _add_model_smoothing(proba_estimator, epsilon=probability_epsilon)
        if (proba_estimator == 0).any() or (proba_estimator == 1.).any():
            raise ValueError("Predictions must not be 0 or 1 for any class")

        log_proba_estimator = np.log(proba_estimator)
        log_proba = log_proba + log_proba_estimator

    return log_proba


def _parallel_predict_poisson_regression(estimators, estimators_features, X):
    """Private function used to compute predictions within a job."""
    return sum(
        np.log(estimator.predict(X[:, features]))
        for estimator, features in zip(estimators, estimators_features)
    )


class BaggingPoissonRegressor(BaseBagging):

    def __init__(
            self,
            base_estimator=None,
            n_estimators=10,
            *,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            warm_start=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator()


    def predict(self, X):

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_poisson_regression)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )
        # Reduce
        y_hat = np.exp(sum(all_y_hat) / self.n_estimators)

        return y_hat

    def _set_oob_score(self, X, y):
        raise NotImplementedError


class GeometricBaggingClassifier(BaggingClassifier):
    """
    sklearn's BaggingClassifier, but predictions and probability estimates are calculated using the normalised geometric
    mean as opposed to the arithmetic mean in sklearn's default implementation.

    Parameters
    ----------
    See Documentation for sklearn's BaggingClassifier for full listing of all parameters omitted here.

    smoothing_factor : float, None (default=None)
        If not None, then the predictions are taken in convex combination with a vector of all 1/n_classes. e.g. for a
        10 class problem, smoothing factor 1e-6 and predictions `x` the smoothed predictions would be `(1 - 1e-6) * x +
        1e-6 * 0.1 * np.oneslike(x)`
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        smoothing_factor=None,
        shared_initial_state=None
    ):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.smoothing_factor = smoothing_factor

        self.shared_initial_state = shared_initial_state

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        all_log_proba = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_predict_log_proba_geometric)(
                self.estimators_[starts[i]: starts[i + 1]],
                self.estimators_features_[starts[i]: starts[i + 1]],
                X,
                self.n_classes_,
                self.smoothing_factor
            )
            for i in range(n_jobs)
        )

        # Reduce
        numerator = np.exp(sum(all_log_proba) /self.n_estimators)
        denominator = numerator.sum(axis=1).reshape((-1, 1))
        proba = numerator / denominator

        return proba

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        This function overwrites the _make_estimator of BaseEnsemble and allows for the creation of ensembles
        where each copy of the base_estimator is created with the same initial state.

        If shared_initial_state is not None, then a new estimator is made with the random state set as (a copy of)
        shared_initial state. If shared_initial_state is None, then _make_estimator from BaseEnsemble is called.

        """
        if self.shared_initial_state is None:
            return super()._make_estimator(append=append, random_state=random_state)

        if self.shared_initial_state is True:
            if not hasattr(self, "estimators_") or len(self.estimators_) == 0:
                self.shared_initial_state = np.random.RandomState(np.random.randint(np.iinfo(np.int32).max))
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})


        member_state = check_random_state(self.shared_initial_state)
        member_state = copy.deepcopy(member_state)
        if random_state is not None:
            _set_random_states(estimator, member_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def _set_oob_score(self, X, y):
        raise NotImplementedError

