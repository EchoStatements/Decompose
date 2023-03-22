import copy
import numbers
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error, zero_one_loss, mean_poisson_deviance
from sklearn.base import is_regressor, is_classifier
from sklearn.ensemble import BaseEnsemble
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm
from tqdm import trange
from decompose import LogisticMarginLoss, ExponentialMarginLoss
from decompose import ZeroOneLoss
from decompose import SquaredLoss, CrossEntropy, PoissonLoss

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class BVDExperiment(object):
    """
    BVDExperiment is the main class used for running experiments where bias, variance and diversity are to be
    calculated, either for a single model configuration or while varying a parameter (either of the ensemble
    or of the base estimators.

    The model, loss and parameter values are passed when the BVDExperiment is initialised, then the training and test
    data are passed to the `run_experiment` function when the experiment is run.

    Examples of usage can be found in `notebooks/Examples_0_Minimal_Examples.ipynb` and `notebooks/Examples_1_Regression_and_Classification`

    Parameters
    ----------
    model : object
        An instance of the model which is to be trained. All parameters should be instantiated to their intended
        values, other than the parameter named in `parameter_name`, whose value will be set by this function
    decomposition_loss : string, optional (default=None)
        If "squared", the experiment expects a regression dataset or if "cross_entropy" a classification dataset is
        expected. For "logistic" and "exponential", binary classification datasets are expected and the bias-variance
        decomposition_class for logistic decomposition_class and margin decomposition_class for exponential decomposition_class are used, respectively.
    parameter_name : string
        The name of the parameter (either an attribute model or model.base_estimator) which is to be varied
        over the course of the experiment
    parameter_values : list
        The list of values which the parameter is to take.
    bootstrap : boolean, optional (default=False)
        If True, when generating data for an individual run of the experiment
        sub-sampling of the training data is done with replacement
    n_samples: float or int, optional (default=0.9)
        The number of samples to draw from the training data to train each base estimator.
           - If int, then draw `n_samples` samples.
           - If float, then draw `n_samples * self.train_data.shape[0]` samples.
    warm_start : boolean, optional (default=None)
        Used for iteratively training the same model(s) while varying the number of training iterations.
        If True, parameter_values should be a monotonically increasing list. For instance, if `parameter_values=[1, 3]`, each model
        is trained for a single iteration, the results evaluated, then each model is trained for a further 2 iterations
         (for a total of three iterations and the results evaluated again. This parameter does not need to be set when
         using sklearn models which implement warm_start, as the value will be inferred from `model`.
    non_centroid_combiner : function
        If None, then the centroid is used. Otherwise, non_centroid_combiner defines a combination rule to be
        used in computing a diversity-like term
    ensemble_warm_start : boolean, optional (default=True)
        Used for iteratively adding members to an ensemble. Should be set to None for sklearn model, with correct behaviour
        inferred from the value of `model.warm_start`.
        If set to True when varying ensemble size, new members
        are added to existing ensembles rather than creating new ensembles from scratch. For instance, if `parameter_name="n_estimators"
        and `parameter_values=[2, 5]`, ensembles of size 2 are trained for the first parameter value; for the second
        parameter value, these ensembles are retained, and 3 more ensemble members are added to each for the ensembles
        of size 5.
        If True, parameter_values should be a monotonically increasing list. Most common use case would be varying
        n_estimators, i.e, the number of estimators in the ensemble.
    compute_zero_one_decomp : boolean, optional (default=False)
        If True, a separate results object is created for the zero one decompose of the loss of models
        trained in the experiment
    save_decompositions: boolean, optional (default=False)
        if True, the results_object created by perform experiment contains the decomposition_class objects, as well as the
        returned values.
    trials_progress_bar : boolean, optional (default=False)
        Determines whether a progress bar is shown for trials within a parameter value, or only progress across
        parameter values is shown
    smoothing_factor : float, optional (default=1e-9)
        When the decomposition_class is specified as "cross_entropy", ensemble members may return a 1-hot vector.
        This will cause the decomposition_class to fail, so instead the class is predicted with probability
        (1 - smoothing_factor), and the remaining probability shared uniformly amongst the other classes.
    log_git_info : boolean
        If True, the logger tries to record information about the current state of the git repository (commit number
        and branch).

    Attributes
    ----------

    results_object : ResultsObject, ZeroOneResultsObject
        Object containing all statistics collected from running `run_experiment`
    all_results: list
        List containing the object returned by run_experiment, and a ZeroOneResultsObject instance if `compute_zero_one_decomp` is True

    """
    def __init__(self,
                 model,
                 loss,
                 parameter_name=None,
                 parameter_values=None,
                 bootstrap=False,
                 n_samples=0.9,
                 warm_start=None,
                 non_centroid_combiner=None,
                 ensemble_warm_start=None,
                 compute_zero_one_decomp=False,
                 save_decompositions=False,
                 decompositions_prefix="decomposition_objects/",
                 trials_progress_bar=False,
                 smoothing_factor=1e-9,
                 per_trial_test_error=False,
                 log_git_info=False):
        logger.debug("Experiment object created")
        if log_git_info:
            _log_git_info()
        self.results_object = None
        self.model = model
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.save_decompositions = save_decompositions
        self.decompositions_prefix = decompositions_prefix
        self.trials_progress_bar = trials_progress_bar
        self.bootstrap = bootstrap
        self.max_samples = n_samples
        self.n_classes = None
        self.n_trials = None
        self.smoothing_factor = smoothing_factor
        self.non_centroid_combiner = non_centroid_combiner
        self.compute_zero_one_decomp = compute_zero_one_decomp

        if self.parameter_values is None:
            self.parameter_values = ["NA"]
        self.is_ensemble = hasattr(self.model, "n_estimators")

        self.decomposition_class, self.loss_func = self._get_decomposition_and_loss_func(loss)
        self._manage_warm_starts(warm_start, ensemble_warm_start)
        logger.debug(f"warm_start: {self.warm_start}, ensemble_warm_start: {self.ensemble_warm_start}")
        self.per_trial_test_error = per_trial_test_error

    def run_experiment(self,
                       train_data,
                       train_labels,
                       test_data,
                       test_labels,
                       train_sample_weight=None,
                       test_sample_weight=None,
                       n_trials=50,
                       n_test_splits=1):
        """
        Performs experiments required to calculate BVD decomposition_class and stores the results.

        Parameters
        ----------
        train_data : ndarray of shape (num_train, num_features)
            The training data for the experiment
        train_labels : ndarray of shape (num_train,)
            The training labels
        test_data : ndarray of shape (num_test, num_features)
            The test data
        test_labels : ndarray of shape (num_test,)
            The test labels
        train_sample_weight : ndarray of shape (num_train,), optional (default=None)
            The weight to be given to each training sample by the loss function
        test_sample_weight : ndarray of shape (num_test,), optional (default=None)
            The weight to be given to each test sample by the loss function
        n_trials : int, optional (default=50)
            The number of runs to be averaged over
        n_test_splits: int, optional (default=1)
            Specifies the number of independent splits test data is separated into. By convention, if we have 2 test
            splits, we treat the first as the validation set and the second as the test set.

        Returns
        -------
        results_object: ResultsObject
            an instance of ResultsObject containing the results from the experiment

        Examples
        --------
        Example usages of this class can be found in the `example_notebooks` directory of this library.

        """

        self.n_trials = n_trials
        self.n_classes = np.unique(train_labels).shape[0]
        n_train = train_data.shape[0]
        n_test = test_data.shape[0]

        if self.decomposition_class in [LogisticMarginLoss, ExponentialMarginLoss, ExponentialMarginLoss,
                                        ZeroOneLoss, CrossEntropy]:
            label_encoder = LabelEncoder()
            label_encoder.fit(train_labels)
            train_labels = label_encoder.transform(train_labels)
            test_labels = label_encoder.transform(test_labels)
            if self.decomposition_class in [LogisticMarginLoss, ExponentialMarginLoss]:
                # Margin losses expect labels in {-1, 1}
                train_labels += -1 * (train_labels == 0 )
                test_labels += -1 * (test_labels == 0 )

        if n_test_splits > 1:
            if test_sample_weight is None:
                test_data, test_labels = shuffle(test_data, test_labels)
            else:
                test_data, test_labels, test_sample_weight = shuffle(test_data, test_labels, test_sample_weight)
        split_test_labels = np.array_split(test_labels, n_test_splits)

        if test_sample_weight is not None:
            split_test_weights = np.array_split(test_sample_weight, n_test_splits)
        else:
            split_test_weights = [None] * n_test_splits

        logger.debug(f"Train examples: {n_train}, Test examples {n_test} in {n_test_splits} splits")

        if self.decomposition_class is not ZeroOneLoss:
            self.results_object = ResultsObject(self.parameter_name, self.parameter_values, self.decomposition_class,
                                                n_test_splits=n_test_splits, save_decompositions=self.save_decompositions,
                                                decompositions_prefix=self.decompositions_prefix)
        else:
            self.results_object = ZeroOneResultsObject(self.parameter_name, self.parameter_values, self.decomposition_class,
                                                       n_test_splits=n_test_splits)

        self.all_results = [self.results_object]

        if self.non_centroid_combiner is not None:
            self.non_centroid_results_object = NonCentroidCombinerResults(self.parameter_name, self.parameter_values,
                                                                          self.decomposition_class,
                                                                          self.non_centroid_combiner,
                                                                          n_test_splits=n_test_splits)
            self.all_results.append(self.non_centroid_results_object)

        if self.compute_zero_one_decomp:
            self.zero_one_results = ZeroOneResultsObject(self.parameter_name, self.parameter_values,
                                                         self.decomposition_class,
                                                         n_test_splits=n_test_splits)
            self.all_results.append(self.zero_one_results)


        # Generate indices for training dataset for each bootstrap
        all_trial_indices = self._generate_bootstraps(train_data,
                                                      replace=self.bootstrap,
                                                      max_samples=self.max_samples)

        logger.debug("Now training: " + str(self.model.__class__.__name__) + ", varying " + str(self.parameter_name))
        logger.debug(f"Model info: {self.model}")
        if not hasattr(self.model, "n_estimators"):
            logger.debug("Note: model does not appear to be an ensemble")

        if self.warm_start or self.ensemble_warm_start:
            models = []

        # assert not hasattr(self.model, "estimators_"), "Estimators should not be defined before fit() is called"

        # Only show progress bar if there is more than one parameter value to iterate through
        if len(self.parameter_values) > 1:
            progress_bar = enumerate(tqdm(self.parameter_values, leave=True))
        else:
            progress_bar = enumerate(self.parameter_values)

        # Iterate through desired parameter values
        for param_idx, param_val in progress_bar:
            # Determine whether model is an ensemble, and if so how many estimators it will have when trained.
            # We do this first so that we know what size to make arrays which store experiment results.
            if self.is_ensemble:
                n_estimators = param_val if self.parameter_name == "n_estimators" else self.model.n_estimators
            else:
                n_estimators = 1

            # Create variables for results of experiments for this parameter value
            this_param_train_error = 0
            this_param_test_error = np.zeros(n_test_splits)

            this_param_member_train_error = 0
            this_param_member_test_error = np.zeros(n_test_splits)
            if self.per_trial_test_error:
                per_trial_test_errors = np.zeros((self.n_trials, n_test_splits))

            if self.decomposition_class == CrossEntropy:
                np_output = np.zeros([n_trials, n_estimators, n_test, self.n_classes])
            elif self.decomposition_class == ZeroOneLoss:
                np_output = np.zeros([n_trials, n_estimators, n_test], dtype=int)
            else:
                np_output = np.zeros([n_trials, n_estimators, n_test])

            if self.compute_zero_one_decomp:
                np_class_pred = np.zeros([n_trials, n_estimators, n_test], dtype=int)

            # Run each trial of the experiment
            range_func = range if not self.trials_progress_bar else trange

            for t_idx in range_func(n_trials):
                logger.debug(f"Current param_idx: {param_idx}, Current trial_idx {t_idx}/{n_trials}")

                # Load training data for this bootstrap
                my_train_data = train_data[all_trial_indices[t_idx], :]
                my_train_labels = train_labels[all_trial_indices[t_idx]]
                if train_sample_weight is not None:
                    my_train_sample_weight = train_sample_weight[all_trial_indices[t_idx]]
                else:
                    my_train_sample_weight = None

                if self.warm_start or self.ensemble_warm_start:
                    # Load model from previous param settings if such a model exists, otherwise we create the model
                    # from the base model
                    if param_idx == 0:
                        cur_model = copy.deepcopy(self.model)
                        models.append(cur_model)
                    else:
                        cur_model = models[t_idx]
                else:
                    cur_model = copy.deepcopy(self.model)

                self._update_model_parameter(cur_model, param_idx)

                self._fit_model(cur_model, my_train_data, my_train_labels, sample_weight=my_train_sample_weight)

                # This line warns if a model finish training before training the desired number of ensemble members
                # as can happen in AdaBoostClassifier if the ensemble perfectly fits the training data.
                if self.is_ensemble and len(cur_model.estimators_) != n_estimators:
                    print(f"Should be {n_estimators} models, are actually {len(cur_model.estimators_)}")
                    logger.warning(f"Should be {n_estimators} models, are actually {len(cur_model.estimators_)}")

                if hasattr(cur_model, "estimators_"):
                    for est_idx, member in enumerate(cur_model.estimators_):
                        np_output[t_idx, est_idx, ...] = self._get_pred(member, test_data)
                        if self.compute_zero_one_decomp:
                            np_class_pred[t_idx, est_idx, ...] = cur_model.estimators_[est_idx].predict(test_data)

                else:
                    assert n_estimators == 1, f"n_estimators should be 1 for single model, not {n_estimators}"
                    np_output[t_idx, 0, ...] = self._get_pred(cur_model, test_data)
                    if self.compute_zero_one_decomp:
                        np_class_pred[t_idx, 0, ...] = cur_model.predict(test_data)

                loss_func = self.loss_func
                # Get predictions from ensemble
                train_preds = cur_model.predict(my_train_data)
                test_preds = cur_model.predict(test_data)
                split_test_preds = np.array_split(test_preds, n_test_splits)

                if self.compute_zero_one_decomp:
                    split_np_class_pred = np.array_split(np_class_pred, n_test_splits, axis=2)

                this_param_train_error += (1 / n_trials) * loss_func(my_train_labels, train_preds,
                                                                   sample_weight=my_train_sample_weight)
                for s_idx in range(n_test_splits):
                    this_test_error = loss_func(split_test_labels[s_idx],
                                           split_test_preds[s_idx],
                                           sample_weight=split_test_weights[s_idx])
                    this_param_test_error[s_idx] += (1 / n_trials) * this_test_error
                    if self.per_trial_test_error:
                        per_trial_test_errors[t_idx, s_idx] = this_test_error

                if hasattr(cur_model, "estimators_") and \
                        (self.parameter_name != "n_estimators" or not self.ensemble_warm_start):
                    avg_member_train_loss, avg_member_test_loss = self._get_individual_errors(cur_model, my_train_data,
                                                                                              my_train_labels, test_data,
                                                                                              test_labels, n_test_splits,
                                                                                              loss_func, train_sample_weight,
                                                                                              test_sample_weight)
                    this_param_member_train_error += (1 / n_trials) * avg_member_train_loss
                    for split_idx in range(n_test_splits):
                        this_param_member_test_error[split_idx] += (1 / n_trials) * avg_member_test_loss[split_idx]

                # If we are varying ensemble size, we save evaluating individual models until the last (largest)
                # parameter value and enter them all into the results_object in a single pass
                if param_idx + 1 == len(self.parameter_values) \
                        and self.parameter_name == "n_estimators" \
                        and self.ensemble_warm_start:
                    logger.debug("Evaluating all member training and test losses at end of run")
                    member_train_losses, member_test_losses = self._get_individual_errors(cur_model, my_train_data,
                                                                                          my_train_labels, test_data,
                                                                                          test_labels,
                                                                                          n_test_splits, loss_func,
                                                                                          return_mean=False)
                    # Fill in the results_object
                    for param_idx2, param_val2 in enumerate(self.parameter_values):
                        for results_object in self.all_results:
                            results_object.member_test_error[param_idx2, :] += (1. / self.n_trials) * member_test_losses[:param_val2,:].mean(axis=0)
                            results_object.member_train_error[param_idx2] += (1. / self.n_trials) * member_train_losses[:param_val2].mean()

            errors = [this_param_train_error, this_param_test_error]
            if self.parameter_name != "n_estimators" or not self.ensemble_warm_start:
                errors += [this_param_member_train_error, this_param_member_test_error]
            if self.per_trial_test_error:
                errors += [per_trial_test_errors]
            # Create decomposition_class objects
            if n_test_splits > 1:
                split_output = np.array_split(np_output, n_test_splits, axis=2)
            else:
                split_output = [np_output]
            for s_idx in range(n_test_splits):
                decomp = self.decomposition_class(split_output[s_idx], split_test_labels[s_idx])

                self.results_object.update_results(decomp, param_idx, errors, split_idx=s_idx,
                                                   sample_weight=split_test_weights[s_idx])
                if self.non_centroid_combiner is not None:
                    self.non_centroid_results_object.update_results(decomp, param_idx, errors, split_idx=s_idx,
                                                                    sample_weight=split_test_weights[s_idx])
                if self.compute_zero_one_decomp:
                    zero_one_decomp = ZeroOneLoss(split_np_class_pred[s_idx], split_test_labels[s_idx])
                    self.zero_one_results.update_results(zero_one_decomp, param_idx, errors, split_idx=s_idx,
                                                         sample_weight=split_test_weights[s_idx])

        return self.results_object


    def _get_individual_errors(self,
                               cur_model,
                               my_train_data,
                               my_train_labels,
                               test_data,
                               test_labels,
                               n_test_splits,
                               loss_func,
                               train_sample_weight=None,
                               test_sample_weight=None,
                               return_mean=True):
        """
        Returns the errors of individual ensemble members on test data and training data. If `return_mean` is True,
        then the average is returned, if not the errors for each individual model are given in a list of length `n_estimators`

        Parameters
        ----------
        cur_model: model
            an ensemble model with a list of an attribute estimators_ containing a list of individual ensemble members
        my_train_data: ndarray of shape (n_samples, n_features)
            The training data used  for the model
        my_train_labels: ndarray of shape (nsamples,)
            The training labels used for the mode
        test_data: ndarray of shape (n_samples, n_features)
            All test data
        test_labels: ndarray of shape (n_samples,)
            All test labels
        n_test_splits: int
            number of disjoint splits that the test data should be split into
        loss_func: func
            the loss function against which the model's performance should be evaluated
        train_sample_weight: ndarray of shape (n_train_samples)
            weighting for each training data point
        test_sample_weight: ndarray of shape (n_test_samples)
            weighting for each test data point
        return_mean: boolean (default=True)
            if True, the average of the losses across all estimators is returned. Otherwise a list of losses for each
            estimator is returned

        Returns
        -------

        member_train_losses : ndarray of size (n_estimators,) or (1,) if `return_mean`
            Loss of ensemble members on the training set
        member_test_losses : ndarray of size (n_estimators, n_test_splits) or (n_test_splits,) if `return_mean`
            Loss of ensemble members on each test split

        """
        n_estimators = len(cur_model.estimators_)
        member_train_losses = np.zeros(n_estimators)
        member_test_losses = np.zeros([n_estimators, n_test_splits])

        split_test_labels = np.array_split(test_labels, n_test_splits)
        if test_sample_weight is not None:
            split_test_weights = np.array_split(test_sample_weight, n_test_splits)
        else:
            split_test_weights = [None] * n_test_splits

        for est_idx, estimator in enumerate(cur_model.estimators_):
            train_preds = estimator.predict(my_train_data)
            test_preds = estimator.predict(test_data)
            split_test_preds = np.array_split(test_preds, n_test_splits)

            member_train_losses[est_idx] += loss_func(my_train_labels, train_preds, sample_weight=train_sample_weight)
            for s_idx in range(n_test_splits):
                member_test_losses[est_idx, s_idx] += loss_func(split_test_labels[s_idx],
                                                          split_test_preds[s_idx],
                                                          sample_weight=split_test_weights[s_idx])
        if return_mean:
            avg_member_train_loss = member_train_losses.mean()
            avg_member_test_loss = member_test_losses.mean(axis=0)
            return avg_member_train_loss, avg_member_test_loss
        else:
            return member_train_losses, member_test_losses


    def _fit_model(self, cur_model, train_data, train_labels, sample_weight=None):
        """
        Helper function to fit model correctly in perform_bvd_experiment

        Parameters
        ----------
        cur_model : object
            The model (or ensemble) to be trained
        train_data : (n_samples, n_features) ndarray
            A numpy array where the ith row contains the features for the ith training example
        train_labels : (n_samples,) ndarray
            A numpy array where the ith row contains the label of the ith example in the training data

        Returns
        -------
        None

        """
        # Running fit() on an ensemble clears the list of ensemble members. In order to warm start ensemble
        # members, it is therefore necessary to iterate through them and fit them individually.
        ensemble_members_exist = hasattr(cur_model, "estimators_")
        if self.warm_start and self.is_ensemble and ensemble_members_exist:
            for member in cur_model.estimators_:
                # This deals with incorrect behaviour of warm_start in MLPClassifier in older versions of sklearn
                if hasattr(member, "classes_"):
                    delattr(member, "classes_")
                print("WARNING: This configuration does not allow models in the ensemble to be passed different data."
                      "To make this work with Bagging, you would need to have member.fit() bootstrap the data that is"
                      "passed to it. This is not implemented in the sklearn standard library.")
                if sample_weight is None:
                    member.fit(train_data, train_labels)
                else:
                    member.fit(train_data, train_labels, sample_weight=sample_weight)
        else:
            if hasattr(cur_model, "classes_"):
                delattr(cur_model, "classes_")
            if sample_weight is None:
                cur_model.fit(train_data, train_labels)
            else:
                cur_model.fit(train_data, train_labels, sample_weight=sample_weight)

        assert hasattr(cur_model, "estimators_") or (not self.is_ensemble), \
            "fit() should generate estimators_ list in ensembles"

    def _update_model_parameter(self, cur_model, param_idx):
        """
        Helper function correctly update model parameters. Looks for parameter in both the ensemble model and in the
        base model, setting whichever is most appropriate.

        Parameters
        ----------
        cur_model: object
            Model whose parameter is to be updated
        param_idx: int
            The position of the parameter value to be used in self.parameter_values
        warm_start: boolean
            If True, updates the parameter to be the difference between the desired parameter value and the current one
            (e.g. if varying number of epochs trained under SGD, we train for the desired number of epochs minus the number
            of epochs for which the model has already been trained.

        Returns
        -------
        None

        """
        # First handle the case where no parameter is being varied
        if self.parameter_name is None:
            return
        # Then handle the case where a parameter is being varied
        param_val = self.parameter_values[param_idx]
        param_update = param_val
        # If warm_start, we want the current param_val to be the difference between the
        # previous desired parameter value and the current parameter value
        # e.g. if previous value was 150 iterations and the current desired value is 200 iterations,
        # we want to train for another 50 iterations.
        if self.warm_start and param_idx > 0:
            param_update = param_val - self.parameter_values[param_idx - 1]
        # Set the parameter's attribute in the ensemble and/or the individual estimators
        param_set = False  # Flag to check that the parameter does actually get set somewhere

        # Deal with the sklearn way of setting parameters
        # This is complicated by the renaming in sklearn of "base_estimator" to "estimator", as a result
        # we have more cases to deal with
        if hasattr(cur_model, "set_params"):
            # If we warm_start ensemble members, we need to reach in and update the parameters in each
            # estimator
            if self.is_ensemble and self.warm_start \
                    and param_idx > 0:
                if self.parameter_name.startswith("base_estimator") or self.parameter_name.startswith("estimator"):
                    _, _, param_name = self.parameter_name.partition("__")
                    for member in cur_model.estimators_:
                        member.set_params(**{param_name: param_update})
                    param_set = True
                else:
                    if hasattr(cur_model, self.parameter_name):
                        cur_model.set_params(**{self.parameter_name: param_update})
                        param_set = True
                    elif hasattr(cur_model, "base_estimator") \
                            and hasattr(cur_model.base_estimator, self.parameter_name):
                        for member in cur_model.estimators_:
                            member.set_params(**{self.parameter_name: param_update})
                        if param_set:
                            logger.warning("WARNING: Setting parameter in ensemble AND in base models")
                        param_set = True
                    # Deal with renaming base_estimator -> estimator
                    elif hasattr(cur_model, "estimator") \
                         and hasattr(cur_model.estimator, self.parameter_name):
                        for member in cur_model.estimators_:
                            member.set_params(**{self.parameter_name: param_update})
                        if param_set:
                            logger.warning("WARNING: Setting parameter in ensemble AND in base models")
                        param_set = True
            else:
                if self.parameter_name in cur_model.get_params():
                    cur_model.set_params(**{self.parameter_name: param_update})
                    param_set = True
                # Unclear whether we want to use hasatttr or `in cur_model.get_params()` here
                elif hasattr(cur_model, "base_estimator") \
                         and hasattr(cur_model.base_estimator, self.parameter_name):
                    cur_model.base_estimator.set_params(**{self.parameter_name: param_update})
                    if param_set:
                        logger.warning("WARNING: Setting parameter in ensemble AND in base models")
                    param_set = True
                elif hasattr(cur_model, "estimator") \
                         and hasattr(cur_model.estimator, self.parameter_name):
                    cur_model.estimator.set_params(**{self.parameter_name: param_update})
                    if param_set:
                        logger.warning("WARNING: Setting parameter in ensemble AND in base models")
                    param_set = True
        # If there is no set_params method, we lose the sklearn way of exactly specifying parameters
        # (e.g. base_estimator__max_tree_depth), so we look for matching parameter names in the base
        # model and the ensemble.
        else:
            if hasattr(cur_model, self.parameter_name):
                setattr(cur_model, self.parameter_name, param_update)
                param_set = True
            if self.is_ensemble and hasattr(cur_model, "base_estimator") and\
                    hasattr(cur_model.base_estimator, self.parameter_name):
                if self.warm_start and param_idx > 0:
                    for member in cur_model.estimators_:
                        assert hasattr(member, self.parameter_name)
                        setattr(member, self.parameter_name, param_update)
                else:
                    setattr(cur_model.base_estimator, self.parameter_name, param_update)
                if param_set:
                    logger.warning("WARNING: Setting parameter in ensemble AND in base models")
                param_set = True
            elif self.is_ensemble and hasattr(cur_model, "estimator") and\
                    hasattr(cur_model.estimator, self.parameter_name):
                if self.warm_start and param_idx > 0:
                    for member in cur_model.estimators_:
                        assert hasattr(member, self.parameter_name)
                        setattr(member, self.parameter_name, param_update)
                else:
                    setattr(cur_model.estimator, self.parameter_name, param_update)
                if param_set:
                    logger.warning("WARNING: Setting parameter in ensemble AND in base models")
                param_set = True
        if not param_set:
            raise RuntimeError("Could not find attribute to set")

    def _get_decomposition_and_loss_func(self, decomp):
        """
        Converts string into decomp class (if necessary) and returns decomp_class and corresponding
        loss function. Needed because it is useful to for the user to be able to set the loss using a string
        but it's also useful to be able to pass an object class (e.g., if implementing a decompose for a custom loss
        function. Additionally, if no loss function is given, it can be inferred from the model being trained.


        Parameters
        ----------
        decomp : str or subclass of Decomposition or None
            Decomposition object to be used or string corresponding to its loss. If None, the loss is inferred from
            `self.model`

        Returns
        -------
        decomp : subclass of Decomposition
            Class with which the decomposition will be calculated.
        loss_func : str
            The name of the loss function as a string.

        """

        # Code for auto-detecting classifier
        if decomp is None:
            if is_regressor(self.model):
                decomp = SquaredLoss
            elif is_classifier(self.model):
                decomp = CrossEntropy
            else:
                raise RuntimeError("Could not identify model as regressor or classifier")

        if isinstance(decomp, str):
            decomp = decomp.lower()
            if decomp == "classify" or decomp == "cross entropy":
                decomp = "cross_entropy"

            lookup = {"cross_entropy": CrossEntropy,
                      "squared": SquaredLoss,
                      "regress": SquaredLoss,
                      "poisson": PoissonLoss,
                      "exponential": ExponentialMarginLoss,
                      "logistic": LogisticMarginLoss,
                      "zero_one": ZeroOneLoss}

            decomp = lookup[decomp]

        if decomp in (CrossEntropy, ExponentialMarginLoss,
                             ZeroOneLoss, LogisticMarginLoss):
            loss_func = zero_one_loss
        elif decomp == SquaredLoss:
            loss_func = mean_squared_error
        elif decomp == PoissonLoss:
            loss_func = mean_poisson_deviance
        else:
            print(f"Decomposition {decomp} not known")
        return decomp, loss_func


    def _get_pred(self, model, data):
        """
        Helper function which gets predictions out of the model. Is necessary in order to allow the introduction of
        prediction noise for classification in a consistent way without duplicating code.

        Parameters
        ----------
        model: object
            The model from which predictions will be collected.

        data : ndarray of shape (n_samples, n_features)
            The input data

        Returns
        -------
        pred: ndarray array of shape (n_samples,)
            The predictions of the model

        """
        if self.decomposition_class == CrossEntropy:
            pred = model.predict_proba(data)
            pred = _add_model_smoothing(pred, epsilon=self.smoothing_factor)
        elif self.decomposition_class in [ExponentialMarginLoss, LogisticMarginLoss]:
            pred = model.decision_function(data)
        else:
            pred = model.predict(data)
        return pred


    def _generate_bootstraps(self, train_data, replace=True, max_samples=1.0):
        """
        Generates a list of ndarrays, each corresponding to the indices used for training the ensemble in a given trial

        Parameters
        ----------
        replace bool, optional (default=True)
            If true, sampling is with replacement, else it without.
        n_samples float, optional (default=1.0)
            The number of samples to draw from X to train each base estimator.
                - If int, then draw `n_samples` samples.
                - If float, then draw `n_samples * X.shape[0]` samples.
            Consistent with the usage of n_samples in sklearn's BaggingClassifier

        Returns
        -------
            all_bootstrap_indices:  list of ndarrays, list is of length self.n_trials
                Each entry in the list is a numpy array listing the indices of the training examples. May contain duplicate indices if sampling training set with
                replacement.

        """
        n_train = train_data.shape[0]
        all_bootstrap_indices = []
        for b_idx in range(self.n_trials):
            if isinstance(max_samples, numbers.Integral):
                # As a special case, if the full dataset is requested we do not use the random choice, instead we just
                # give all indices in ascending order
                if max_samples == n_train and replace is False:
                    sample_indices = np.arange(n_train)
                else:
                    sample_indices = np.random.choice(n_train, max_samples, replace=replace)
            elif isinstance(max_samples, numbers.Real):
                if max_samples == 1.0 and replace == False:
                    # Deal with the full dataset special case again
                    sample_indices = np.arange(n_train)
                else:
                    sample_indices = np.random.choice(n_train, int(max_samples * n_train),
                                                      replace=replace)
            else:
                raise ValueError("Data type of n_samples not understood, datatype is {type(n_samples)}",
                                 "expected float or int")
            all_bootstrap_indices.append(sample_indices)
        return all_bootstrap_indices


    def _manage_warm_starts(self, warm_start, ensemble_warm_start):
        """
        If warm_start and ensemble_warm_start are set as function arguments, they are always deferred to, otherwise
        we attempt to infer the optimal warm_start behaviour from the model/ensemble

        Parameters
        ----------
        warm_start : boolean
        ensemble_warm_start : boolean

        Returns
        -------
        None

        """
        # We check that the behaviour of the ensemble is what the user desires by comparing the warm start
        # attributes of the ensemble and base estimator against the arguments passed to BVDExperiment.

        # Code for auto-detecting warm_start

        # 1st case is when the model is an ensemble with warm-starting members, second case is when the model
        # is _not_ an ensemble, but does have a warm start attribute
        if hasattr(self.model, "base_estimator") and hasattr(self.model.base_estimator, "warm_start"):
            inferred_warm_start = self.model.base_estimator.warm_start
        elif hasattr(self.model, "estimator") and hasattr(self.model.estimator, "warm_start"):
            inferred_warm_start = self.model.estimator.warm_start
        elif not hasattr(self.model, "base_estimator") and not hasattr(self.model, "estimator") and hasattr(self.model, "warm_start"):
            inferred_warm_start = self.model.warm_start
        else:
            inferred_warm_start = None

        # Deal with disagreement and failed detection
        if warm_start is not None:
            self.warm_start = warm_start
            if inferred_warm_start is not None and inferred_warm_start != warm_start:
                print("WARNING: Disagreement between warm_start and the value of inferred_warm_start from the model")
        else:
            if inferred_warm_start is not None:
                self.warm_start = inferred_warm_start
            else:
                self.warm_start = False

        # auto-detect ensemble_warm_start
        if isinstance(self.model, BaseEnsemble) and hasattr(self.model, "warm_start"):
            inferred_ensemble_warm_start = self.model.warm_start
        else:
            inferred_ensemble_warm_start = None

        # deal with disagreement and failed detection
        if ensemble_warm_start is not None:
            self.ensemble_warm_start = ensemble_warm_start
            if inferred_ensemble_warm_start is not None and ensemble_warm_start != inferred_ensemble_warm_start:
                print("WARNING: Disagreement between ensemble_warm_start and the value of inferred_ensemble_warm_start",
                      "inferred from the model")
        else:
            if inferred_ensemble_warm_start is not None:
                self.ensemble_warm_start = inferred_ensemble_warm_start
            else:
                self.ensemble_warm_start = False

        # The following deals with cases where the user has provided a warm_start keyword, but the model
        # configuration does not make sense.
        if self.ensemble_warm_start and not self.parameter_name == "n_estimators":
            print("Warning: Using warm_start in ensemble, but not varying n_estimators may",
                  " result in no models being trained")
        if self.warm_start and self.ensemble_warm_start:
            raise RuntimeError("both warm_start and ensemble_warm_start are turned on. This is not a valid experimental"
                               " configuration.")

        if hasattr(self.model, "warm_start") \
                and not self.is_ensemble \
                and self.model.warm_start != self.warm_start:
            print("WARNING: Disagreement between warm_start arguments in self.model and experiment configuration.")

        if hasattr(self.model, "base_estimator") \
                and hasattr(self.model.base_estimator, "warm_start") \
                and self.model.base_estimator.warm_start != self.warm_start:
            print("WARNING: Disagreement between warm_start arguments in base_estimator and experiment configuration.")

        if hasattr(self.model, "estimator") \
                and hasattr(self.model.base_estimator, "warm_start") \
                and self.model.estimator.warm_start != self.warm_start:
            print("WARNING: Disagreement between warm_start arguments in estimator and experiment configuration.")


def load_results(file_name):
    """
    Function to load in ResultsObjects (or ZeroOneResultsObjects, etc) which have been saved to disk.

    Parameters
    ----------
    file_name : str
        The name of the pickle file containing the results object.

    Returns
    -------
    results : ResultsObject or similar
        The results object loaded from the pickle file

    """

    with open(file_name, "rb") as file_:
        results = pickle.load(file_)
    return results


class ResultsObject(object):
    """
    Results from BVDExperiment are stored in ResultsObject instances.

    Parameters
    ----------
    parameter_name : str
        name of parameter that is varied over the course of the experiment
    parameter_values : list
        list of values that the varied parameter takes
    loss_func : str
        name of the loss function used for the decompose
    n_test_splits : int
        Number of separate folds unseen data is split into. Default is 2, with the first being the train split and the
        second being test
    save_decompositions : boolean
        If True, the decompose objects used in calculating the statistics stored in the results object are also saved.
        This is used if we also want to save the predictions and labels along with statistics.
    decomposition_prefix : str
        Used to give the prefix (including directory) for the file names that we want to use to store the decompose
        objects

    Attributes
    ----------
    ensemble_risk : ndarray of shape (n_parameter_values, n_test_splits)
        The risk of the ensemble for each parameter value and test split
    ensemble_bias: ndarray of shape (n_parameter_values, n_test_splits)
        The biasof the ensemble for each paramter value and test split
    ensemble_variance: ndarray of shape (n_parameter_values, n_test_splits)
        The varianceof the ensemble for each parameter value and test split
    average_bias : ndarray of shape (n_parameter_values, n_test_splits)
        The average bias of the ensemble members for each parmater value and test split
    average_variance : ndarray of shape (n_parameter_values, n_test_splits)
        The average variance of the ensemble members for each parmater value and test split
    diversity : ndarray of shape (n_parameter_values, n_test_splits)
        The diversity for each parameter value and test split
    test_error : ndarray of shape (n_parameter_values, n_test_splits)
        The test error of the ensemble for each parameter value and test split
    train_error : ndarray of shape (n_parameter_values)
        The train error of the ensemble for each parameter value (each ensemble is evaluated only on data that it has seen
        during training.
    member_test_error : ndarray of shape (n_parameter_values, n_test_splits)
        The average test error of the ensemble for each parameter value and test split
    member_train_error : ndarray of shape (n_parameter_values)
        The average train error of the ensemble for each parameter value. Members are evaluated on data that was seen by the
        ensemble during training; there may be examples that were seen by the ensemble but not the individual member

    """

    def __init__(self, parameter_name, parameter_values, loss_func,
                 n_test_splits, save_decompositions=False, decompositions_prefix="decomposition_object_names/"):
        n_parameter_values = len(parameter_values)
        self.loss_func = loss_func
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.n_test_splits = n_test_splits
        self.ensemble_risk = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_bias = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_variance = np.zeros((n_parameter_values, n_test_splits))
        self.average_bias = np.zeros((n_parameter_values, n_test_splits))
        self.average_variance = np.zeros((n_parameter_values, n_test_splits))
        self.diversity = np.zeros((n_parameter_values, n_test_splits))
        self.test_error = np.zeros((n_parameter_values, n_test_splits))
        self.train_error = np.zeros((n_parameter_values))
        self.member_test_error = np.zeros((n_parameter_values, n_test_splits))
        self.member_train_error = np.zeros((n_parameter_values))

        self.save_decompositions = save_decompositions
        self.decomposition_prefix = decompositions_prefix
        if save_decompositions:
            self.decomposition_object_names = [[] for _ in range(n_test_splits)]

    def update_results(self, decomp, param_idx, errors, split_idx=0, sample_weight=None):
        """
        Function used to update ResultsObject for a new parameter using Decomposition object and list of train/test errors

        Parameters
        ----------
        decomp : Decomposition
            Decomposition object for the experiment
        param_idx : int
            The index of the current parameter in the parameter_values
        errors : list of floats
            List containing (in order):
                Training error averaged over all runs of the experiment
                Test error averaged over all runs of the experiment
                (optional)
                Average member train error
                Average member test error

        Returns
        -------
        None

        """
        self.train_error[param_idx] = errors[0]  # overall error
        self.test_error[param_idx, split_idx] = errors[1][split_idx]  # overall error

        if len(errors) == 4:
            self.member_train_error[param_idx] = errors[2]  # avg member error
            self.member_test_error[param_idx, split_idx] = errors[3][split_idx]  # avg member error

        # We can tell if we have per trial test erorrs by checking if the length of the errors list is odd
        if len(errors) % 2 == 1:
            if not hasattr(self, "per_trial_test_errors"):
                self.per_trial_test_errors = np.zeros((len(self.parameter_values),
                                                       errors[-1].shape[0],
                                                       self.n_test_splits))
            # This also doesn't feel great, is it already filled?
            self.per_trial_test_errors[param_idx, :, split_idx] = errors[-1][:, split_idx]


        self.ensemble_bias[param_idx, split_idx] = np.average(decomp.ensemble_bias,
                                                              weights=sample_weight)

        self.ensemble_variance[param_idx, split_idx] = np.average(decomp.ensemble_variance,
                                                                                  weights=sample_weight)

        self.average_bias[param_idx, split_idx] = np.average(decomp.average_bias,
                                                             weights=sample_weight)

        self.average_variance[param_idx, split_idx] = np.average(decomp.average_variance,
                                                                 weights=sample_weight)

        self.diversity[param_idx, split_idx] = np.average(decomp.diversity, weights=sample_weight)

        self.ensemble_risk[param_idx, split_idx] = np.average(decomp.expected_ensemble_loss,
                                                              weights=sample_weight)
        logger.debug(f"Update Summary {param_idx},{split_idx}--"
                     f"ensemble bias: {self.ensemble_bias[param_idx, split_idx]},"
                     f" ensemble variance: {self.ensemble_variance[param_idx, split_idx]},"
                     f" average bias: {self.average_bias[param_idx, split_idx]},"
                     f"average variance: {self.average_variance[param_idx, split_idx]}, "
                     f"diversity: {self.diversity[param_idx, split_idx]},"
                     f" ensemble risk{self.ensemble_risk[param_idx, split_idx]},"
                     f" test error:{self.test_error[param_idx, split_idx]},"
                     f" train error{self.train_error[param_idx]}")

        if self.save_decompositions:
            # Decomposition objects can get prohibitively large, so we save them to hard drive and just
            # keep a list of file names for when they need to be restored
            if not self.parameter_name == "n_estimators" or param_idx == len(self.parameter_values) - 1:
                # When updating the ensemble size, we don't need to save for every parameter index, since
                # we can cheaply reconstruct smaller ensembles from the largest one
                decomposition_filename = f"{self.decomposition_prefix}_{param_idx}_{split_idx}.pkl"
                self.decomposition_object_names[split_idx].append(decomposition_filename)
                with open(decomposition_filename, "wb") as file_:
                    pickle.dump(decomp, file_)
                    logger.debug(f"writing decompose object to {decomposition_filename}")

    def print_summary(self, splits=[0]):
        """
        Prints summary of available statistics regarding the decomposition_class, as inferred from experimental results.

        Parameters
        ----------
        splits : int or list of ints
            List of test splits for which statistics are to be printed

        """
        print("Average bias:")
        print(self.average_bias[:, splits])
        print("\nAverage variance:")
        print(self.average_variance[:, splits])
        print("\nDiversity:")
        print(self.diversity[:, splits])
        print("\nEnsemble expected risk:")
        print(self.ensemble_risk[:, splits])

    def get_bvd_terms(self, test_splits=[0]):
        """
        Function that returns bias, variance and diversity for given test splits

        Parameters
        ----------
        splits : int or list of ints
            List of test splits for which statistics are to be printed

        Returns
        -------
        bias : ndarray of shape (len(test_splits,))
            The average bias for each test split in `test_splits`
        variance : ndarray of shape (len(test_splits,))
            The average variance for each test split in `test_splits`
        diversity : ndarray of shape (len(test_splits,))
            The diversity for each test split in `test_splits`

        """
        if not hasattr(self, "n_trials"):
            print("WARNING: run_experiment() need to execute before decomposition_class terms are available")

        if np.allclose(self["average_variance"], self["ensemble_variance"]):
            print("WARNING: Ensemble variance and average variance are the same, this suggests that the model"
                  "_is not_ an ensemble, and get_bv_terms() should be used instead.")
        bias = self["average_bias"][:, test_splits]
        variance = self["average_variance"][:, test_splits]
        diversity = self["diversity"][:, test_splits]
        return bias, variance, diversity

    def get_risk(self, test_splits=[0]):
        """
        Returns the ensemble risk for given test splits

        Parameters
        ----------
        test_splits : int or list of ints
            List of test splits for which statistics are to be returned

        Returns
        -------
        ensemble_risk : ndarray of shape (len(test_splits),)

        """
        if not hasattr(self, "n_trials"):
            print("WARNING: run_experiment() need to execute before decomposition_class terms are available")
        risk = self["ensemble_risk"][:, test_splits]
        return risk

    def get_bv_terms(self, test_splits=[0]):
        """

        Parameters
        ----------
        test_splits : int or list of ints
            List of test splits for which statistics are to be returned

        Returns
        -------
            List of test splits for which statistics are to be returned

        """
        if not hasattr(self, "n_trials"):
            print("WARNING: run_experiment() need to execute before decomposition_class terms are available")
        bias = self["ensemble_bias"][:, test_splits]
        variance = self["ensemble_variance"][:, test_splits]
        return bias, variance

    def get_errors(self, test_splits=None, return_avg_member_errors=False):
        if test_splits is None:
            test_splits = 1 if self["test_error"].shape[1] > 1 else 0
        if not hasattr(self, "n_trials"):
            print("WARNING: run_experiment() need to execute before decomposition_class terms are available")
        train_errors = self["train_error"]
        test_errors = self["test_error"][:, test_splits]
        errors = [train_errors, test_errors]
        if return_avg_member_errors:
            member_train_errors = self["member_train_error"]
            member_test_errors = self["member_train_error"]
            errors.append(member_train_errors, member_test_errors)
        return errors

    def get_decomposition_object(self, param_idx, split_idx):
        """
        Gets decompose object for a given parameter index and test split idx

        Parameters
        ----------
        param_idx : int
            index of parameter (in parameter_values)
        split_idx : int
            index of test split

        Returns
        -------
        decomp_object : Decomposition
            The decompose object for unseen data in the split corresponding to `split_idx` for the parameter value
            corresponding to `param_idx`

        """

        if not self.parameter_name == "n_estimators":
            decomposition_filename = f"{self.decomposition_prefix}_{param_idx}_{split_idx}.pkl"
            with open(decomposition_filename, "rb") as decomp_file:
                decomp_object = pickle.load(decomp_file)
            return decomp_object
        else:
            decomposition_filename = f"{self.decomposition_prefix}_{len(self.parameter_values) - 1}_{split_idx}.pkl"
            with open(decomposition_filename, "rb") as decomp_file:
                decomp_object = pickle.load(decomp_file)
            # create new decompose object
            # we do this so as to not have to worry about cached_properties being saved from larger pred size
            decomp_object.pred = decomp_object.pred[:, :self.parameter_values[param_idx], ...]
            DecompClass = decomp_object.__class__
            new_decomp_object = DecompClass(decomp_object.pred, decomp_object.labels)
            return new_decomp_object

    def save_results(self, file_name):
        """
        Saves results object to pickle file for later use

        Parameters
        ----------
        file_name : str
            name of file (inlcuding directory) in which results are toe be stored

        Returns
        -------
        None

        """
        with open(file_name, "wb+") as file_:
            pickle.dump(self, file_)
            logger.debug(f"Writing results to {file_name}")


class NonCentroidCombinerResults(object):
    """
    Used for experiments where the ensemble combiner is not the centroid combiner rule. This feature is functioning
    though not as well tested as other decompositions

    """

    def __init__(self, parameter_name, parameter_values, loss_func, non_centroid_combiner,
                 n_test_splits):
        n_parameter_values = len(parameter_values)
        self.loss_func = loss_func
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.non_centroid_combiner = non_centroid_combiner
        self.ensemble_risk = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_bias = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_variance = np.zeros((n_parameter_values, n_test_splits))
        self.average_bias = np.zeros((n_parameter_values, n_test_splits))
        self.average_variance = np.zeros((n_parameter_values, n_test_splits))
        self.diversity_like = np.zeros((n_parameter_values, n_test_splits))
        self.target_dependent_term = np.zeros((n_parameter_values, n_test_splits))
        self.test_error = np.zeros((n_parameter_values, n_test_splits))
        self.train_error = np.zeros((n_parameter_values))
        self.member_test_error = np.zeros((n_parameter_values, n_test_splits))
        self.member_train_error = np.zeros((n_parameter_values))

    def update_results(self, decomp, param_idx, errors, split_idx=0, sample_weight=None):
        """

        Parameters
        ----------
        decomp : Decomposition
            Decomposition object for the experiment
        param_idx : int
            The index of the current parameter in the parameter_values
        errors : list of floats
            Training error averaged over all runs of the experiment
            Test error averaged over all runs of the experiment
            (optional)
            Average member train error
            Average member test error

        Returns
        -------

        """
        self.train_error[param_idx] = errors[0]  # overall error
        self.test_error[param_idx, split_idx] = errors[1][split_idx]  # overall error

        if len(errors) == 4:
            self.member_train_error[param_idx] = errors[2]  # avg member error
            self.member_test_error[param_idx, split_idx] = errors[3][split_idx]  # avg member error

        self.ensemble_bias[param_idx, split_idx] = np.average(decomp.ensemble_bias,
                                                              weights=sample_weight)

        self.ensemble_variance[param_idx, split_idx] = np.average(decomp.ensemble_variance,
                                                                  weights=sample_weight)

        self.average_bias[param_idx, split_idx] = np.average(decomp.average_bias,
                                                             weights=sample_weight)

        self.average_variance[param_idx, split_idx] = np.average(decomp.average_variance,
                                                                 weights=sample_weight)

        self.diversity_like[param_idx, split_idx] = np.average(decomp.diversity_like(self.non_centroid_combiner),
                                                               weights=sample_weight)

        self.target_dependent_term[param_idx, split_idx] = np.average(decomp.target_dependent_term(self.non_centroid_combiner),
                                                                      weights=sample_weight)

        self.ensemble_risk[param_idx, split_idx] = np.average(decomp.non_centroid_expected_ensemble_risk(self.non_centroid_combiner),
                                                              weights=sample_weight)


class ZeroOneResultsObject(object):
    """
    Results object for the results of experiments examining the effect decompose and the 0-1 loss

    Parameters
    ----------
    parameter_name : str
        name of parameter that is varied over the course of the experiment
    parameter_values : list
        list of values that the varied parameter takes
    loss_func : str
        name of the loss function used for the decompose
    n_test_splits : int
        Number of separate folds unseen data is split into. Default is 2, with the first being the train split and the
        second being test

    Attributes
    ----------
    ensemble_risk : ndarray of shape (n_parameter_values, n_test_splits)
        The risk of the ensemble for each parameter value and test split
    ensemble_bias_effect : ndarray of shape (n_parameter_values, n_test_splits)
        The bias-effect of the ensemble for each paramter value and test split
    ensemble_variance_effect : ndarray of shape (n_parameter_values, n_test_splits)
        The variance-effect of the ensemble for each parameter value and test split
    average_bias_effect : ndarray of shape (n_parameter_values, n_test_splits)
        The average bias-effect of the ensemble members for each parmater value and test split
    average_variance_effect : ndarray of shape (n_parameter_values, n_test_splits)
        The average variance of the ensemble members for each parmater value and test split
    diversity_effect : ndarray of shape (n_parameter_values, n_test_splits)
        The diversity-effect for each parameter value and test split
    test_error : ndarray of shape (n_parameter_values, n_test_splits)
        The test error of the ensemble for each parameter value and test split
    train_error : ndarray of shape (n_parameter_values)
        The train error of the ensemble for each parameter value (each ensemble is evaluated only on data that it has seen
        during training.
    member_test_error : ndarray of shape (n_parameter_values, n_test_splits)
        The average test error of the ensemble for each parameter value and test split
    member_train_error : ndarray of shape (n_parameter_values)
        The average train error of the ensemble for each parameter value. Members are evaluated on data that was seen by the
        ensemble during training; there may be examples that were seen by the ensemble but not the individual member

    See Also
    --------
    ResultsObject

    """

    def __init__(self, parameter_name, parameter_values, loss_func,
                 n_test_splits):
        n_parameter_values = len(parameter_values)
        self.loss_func = loss_func
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.ensemble_risk = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_bias_effect = np.zeros((n_parameter_values, n_test_splits))
        self.ensemble_variance_effect = np.zeros((n_parameter_values, n_test_splits))
        self.average_bias_effect = np.zeros((n_parameter_values, n_test_splits))
        self.average_variance_effect = np.zeros((n_parameter_values, n_test_splits))
        self.diversity_effect = np.zeros((n_parameter_values, n_test_splits))
        self.test_error = np.zeros((n_parameter_values, n_test_splits))
        self.train_error = np.zeros((n_parameter_values))
        self.member_test_error = np.zeros((n_parameter_values, n_test_splits))
        self.member_train_error = np.zeros((n_parameter_values))

    def update_results(self, decomp, param_idx, errors, split_idx=0, sample_weight=None):
        """

        Parameters
        ----------
        decomp : Decomposition
            Decomposition object for the experiment
        param_idx : int
            The index of the current parameter in the parameter_values
        errors : list of floats
            Training error averaged over all runs of the experiment
            Test error averaged over all runs of the experiment
            (optional)
            Average member train error
            Average member test error

        Returns
        -------
        None

        """
        self.train_error[param_idx] = errors[0]  # overall error
        self.test_error[param_idx, split_idx] = errors[1][split_idx]  # overall error

        if len(errors) == 4:
            self.member_train_error[param_idx] = errors[2]  # avg member error
            self.member_test_error[param_idx, split_idx] = errors[3][split_idx]  # avg member error

        self.ensemble_risk[param_idx, split_idx] = np.average(decomp.expected_ensemble_loss,
                                                              weights=sample_weight)
        self.ensemble_bias_effect[param_idx, split_idx] = np.average(decomp.average_bias,
                                                                     weights=sample_weight)

        self.ensemble_variance_effect[param_idx, split_idx] = np.average(decomp.ensemble_variance_effect,
                                                                         weights=sample_weight)

        self.average_bias_effect[param_idx, split_idx] = np.average(decomp.average_bias,
                                                                    weights=sample_weight)

        self.average_variance_effect[param_idx, split_idx] = np.average(decomp.average_variance_effect,
                                                                        weights=sample_weight)

        self.diversity_effect[param_idx, split_idx] = np.average(decomp.diversity_effect, weights=sample_weight)



def _add_model_smoothing(pred, epsilon=1e-9):
    """
    Takes a 2-d numpy array of size N x K (number of examples by number of classes) and adds label noise. Takes
    an array with class probabilities for each example, increases the probability of each class by a small amount
    (epsilon), then re-normalises to ensure a valid probability distribution.

    Parameters
    ----------
    pred : numpy ndarray of shape (num_samples, num_labels)
        Predictions to which noise is to be added
    epsilon : float, optional (default=1e-9)
        size of the (unnormalized) increase in probability for each class

    Returns

    -------

    pred : numpy ndarray of shape (num_samples x num_labels)
        Predictions with added noise
    """
    if epsilon in [0., None]:
        return pred
    else:
        if epsilon < 0 or epsilon > 1.:
            raise ValueError("Value between 0 and 1 expected for smoothing factor")
        return (1 - epsilon) * pred + epsilon * (np.ones_like(pred)) * (1. / pred.shape[1])

def _log_git_info():
    """
    Logs information about git commit and branch

    Returns
    -------
    None

    """
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        head_obj = repo.head.object
        git_info = {
            "commit": head_obj.hexsha,
            "branch": repo.active_branch.name
        }
        logger.info(f"git commit info: {git_info}")
    except ModuleNotFoundError:
        logger.warning("Could not find git module, not logging information about git commit")
    except git.exc.InvalidRepositoryError:
        logger.warning("Experiments file does not appear to be in git repository, failed to log git information")
    except Exception:
        logger.warning("Unexpected Error has occured when attempting to log git information")
