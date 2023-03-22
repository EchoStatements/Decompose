import unittest
import numpy as np
import copy
from numpy.testing import assert_allclose
from scipy.stats import dirichlet
from test_classes import (Gaussian, MultivariateGaussian, Bernoulli,
                           Categorical, Poisson)

from test_classes import  unweighted_mode
from decompose.zero_one_loss import mode_with_random_ties, ZeroOneLoss

from decompose import (SquaredLoss, MultivariateSquaredLoss, BinaryCrossEntropy,
                       CrossEntropy, PoissonLoss, BujaExponentialLoss, LogisticMarginLoss)
from decompose.experiments import _add_model_smoothing

from decompose.experiments import BVDExperiment

class LossesTests(unittest.TestCase):

    def setUp(self):
        self.models_with_diversity = []
        self.other_models = []

        D = np.random.randint(10, 20)  # number of training datasets
        M = np.random.randint(10, 20)  # number of ensemble members
        N = np.random.randint(200, 300)  # number of test examples
        K = np.random.randint(10, 20)  # dimensionality of dependent variable

        # Squared Loss
        data = np.random.normal(size=(D, M, N))
        labels = np.random.normal(size=N)
        self.models_with_diversity.append(SquaredLoss(data, labels))

        # Multivariate Squared Loss
        data = np.random.normal(size=(D, M, N, K))
        labels = np.random.normal(size=(N, K))
        self.models_with_diversity.append(MultivariateSquaredLoss(data, labels))

        # Binary Crossy Entropy
        data = np.random.random(size=(D, M, N))
        labels = np.random.choice(2, size=N)
        self.models_with_diversity.append(BinaryCrossEntropy(data, labels))

        # Cross Entropy
        data = dirichlet([1]*K).rvs((D, M, N))
        labels = np.random.choice(K, size=N)
        self.models_with_diversity.append(CrossEntropy(data, labels))

        # Poisson Loss
        data = np.random.uniform(0, 10, size=(D, M, N))
        labels = np.random.choice(20, size=N)
        self.models_with_diversity.append(PoissonLoss(data, labels))

        # ExponentialLoss
        data = np.random.normal(size=(D, M, N))
        labels = np.random.choice((-1, 1), size=N)
        self.other_models.append(BujaExponentialLoss(data, labels))

        # LogisticMarginLoss
        self.other_models.append(LogisticMarginLoss(data, labels))

    def test_grad_inverse_composition(self):
        for model in self.models_with_diversity + self.other_models:
            if hasattr(model, "_inverse_generator_gradient"):
                pred2 = model._inverse_generator_gradient(model.etas)
                assert_allclose(model.pred, pred2)

    def test_ensemble_bias_variance_decomposition(self):
        for model in self.models_with_diversity + self.other_models:
            LHS = model.expected_ensemble_loss
            RHS = model.ensemble_bias + model.ensemble_variance
            assert_allclose(LHS, RHS)

    def test_average_bias_variance_decomposition(self):
        for model in self.models_with_diversity:
            LHS = model.expected_member_loss
            RHS = model.average_bias + model.average_variance
            assert_allclose(LHS, RHS)

    def test_ensemble_bias_decomposition(self):
        for model in self.models_with_diversity:
            LHS = model.ensemble_bias
            RHS = model.average_bias - model.disparity
            assert_allclose(LHS, RHS)

    def test_ensemble_variance_decomposition(self):
        for model in self.models_with_diversity + self.other_models:
            LHS = model.ensemble_variance
            RHS = (model.average_variance + model.disparity
                   - model.diversity)
            assert_allclose(LHS, RHS)

    def test_full_decomposition(self):
        for model in self.models_with_diversity:
            LHS = model.expected_ensemble_loss
            RHS = (model.average_bias
                   + model.average_variance
                   - model.diversity)
            assert_allclose(LHS, RHS)

    def test_non_centroid_decomposition(self):
        combination_function = np.mean
        for model in self.models_with_diversity:
            LHS = model.non_centroid_expected_ensemble_risk(combination_function)
            RHS = (model.average_bias
                   + model.average_variance
                   - model.diversity_like(combination_function)
                   - model.target_dependent_term(combination_function))
            assert_allclose(LHS, RHS)
            print(f"{model} passed")


class EffectDecompositionTests(unittest.TestCase):

    def setUp(self):
        self.models = []

        D = np.random.randint(10, 20)  # number of training datasets
        M = np.random.randint(10, 20)  # number of ensemble members
        N = np.random.randint(10, 20)  # number of test examples
        K = np.random.randint(10, 20)  # dimensionality of dependent variable

        # Gaussian
        data = np.random.choice(K, size=(D, M, N))
        labels = np.random.choice(K, size=N)
        self.models.append(ZeroOneLoss(data, labels, random_ties=True))

    def test_zero_one_diversity(self):
        for model in self.models:
            LHS = model.expected_ensemble_loss
            RHS = model.average_bias + model.average_variance_effect - model.diversity_effect
            assert_allclose(LHS, RHS)




class DistributionTests(unittest.TestCase):

    def setUp(self):
        self.models = []

        D = np.random.randint(10, 20)  # number of training datasets
        M = np.random.randint(10, 20)  # number of ensemble members
        N = np.random.randint(10, 20)  # number of test examples
        K = np.random.randint(10, 20)  # dimensionality of dependent variable

        # Gaussian
        data = np.random.normal(size=(D, M, N))
        labels = np.random.normal(size=N)
        self.models.append(Gaussian(data, labels))

        # Multivariate Gaussian
        data = np.random.normal(size=(D, M, N, K))
        labels = np.random.normal(size=(N, K))
        self.models.append(MultivariateGaussian(data, labels))

        # Bernoulli
        data = np.random.random(size=(D, M, N))
        labels = np.random.choice(2, size=N)
        bernoulli = Bernoulli(data, labels)
        self.models.append(bernoulli)

        # Categorical
        data = dirichlet([1]*K).rvs((D, M, N))
        labels = np.random.choice(K, size=N)
        categorical = Categorical(data, labels)
        self.models.append(categorical)

        # Poisson
        data = np.random.uniform(0, 10, size=(D, M, N))
        labels = np.random.choice(20, size=N)
        self.models.append(Poisson(data, labels))

    def test_link_inverse_link_composition(self):
        for model in self.models:
            pred2 = model._canonical_inverse_link(model.etas)
            assert_allclose(model.pred, pred2)

    def test_ensemble_bias_variance_decomposition(self):
        for model in self.models:
            LHS = model.expected_ensemble_risk
            RHS = model.ensemble_central_pred_risk + model.ensemble_variance
            assert_allclose(LHS, RHS)

    def test_average_bias_variance_decomposition(self):
        for model in self.models:
            LHS = model.expected_average_model_risk
            RHS = model.average_central_risk + model.average_model_variance
            assert_allclose(LHS, RHS)

    def test_ensemble_bias_decomposition(self):
        for model in self.models:
            LHS = model.ensemble_central_pred_risk
            RHS = model.average_central_risk - model.disparity
            assert_allclose(LHS, RHS)

    def test_ensemble_variance_decomposition(self):
        for model in self.models:
            LHS = model.ensemble_variance
            RHS = (model.average_model_variance + model.disparity
                   - model.diversity)
            assert_allclose(LHS, RHS)

    def test_full_decomposition(self):
        for model in self.models:
            LHS = model.expected_ensemble_risk
            RHS = (model.average_central_risk
                   + model.average_model_variance
                   - model.diversity)
            assert_allclose(LHS, RHS)

    def test_gaussian_nll(self):
        # TODO: integrate better with variables in setup
        labels = self.models[0].labels
        pred = self.models[0].labels.reshape((1, 1, -1))
        pred = np.tile(pred, (10, 10, 1))
        nll = self.models[0]._compute_NLL(pred, labels)
        assert_allclose(nll, 0) # np.log(2 * np.pi) / 2)
        nll = self.models[0]._compute_NLL(pred + 1, labels)
        assert_allclose(nll, 0.5) #np.log(2 * np.pi) / 2 + 0.5)


from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble

"""
BVD Experiments now has two behaviours for parameter update, one for sklearn models, where it uses the internal
set_params() method and an one for generic ML models, where it uses the parameter_name. 
"""

class SKLearnTestModel(BaseEstimator):

    def __init__(self, model_parameter=0, shared_parameter=0, nested_model=False):
        self.n_times_fit_called = 0
        self.model_parameter = model_parameter
        self.shared_parameter = shared_parameter
        if nested_model:
            self.nested_model = SKLearnTestModel(nested_model=False)
        else:
            self.nested_model = None

    def fit(self, X, y):
        self.n_times_fit_called += 1


class SKLearnTestEnsemble(BaseEnsemble):

    def __init__(self, base_estimator=None, n_estimators=5, ensemble_parameter=0, shared_parameter=0):
        self.n_estimators = n_estimators
        self.n_times_fit_called = 0
        self.ensemble_parameter = ensemble_parameter
        self.shared_parameter = shared_parameter
        if base_estimator is None:
            self.base_estimator = SKLearnTestModel()
        else:
            self.base_estimator = base_estimator


    def fit(self, X, y):
        # In application, the deep copy here would be done by sklearn's clone() function
        self.estimators_ = [copy.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        self.n_times_fit_called += 1

class TestModel(object):

    def __init__(self):
        self.n_times_fit_called = 0
        self.model_parameter = 0
        self.shared_parameter = 0

    def fit(self, X, y):
        self.n_times_fit_called += 1


class TestEnsemble(object):

    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.n_times_fit_called = 0
        self.ensemble_parameter = 0
        self.shared_parameter = 0
        self.base_estimator = TestModel()

    def fit(self, X, y):
        # In application, the deep copy here would be done by sklearn's clone() function
        self.estimators_ = [copy.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        self.n_times_fit_called += 1

class BVDExperimentTests(unittest.TestCase):

    def setUp(self):
        self.bvd_experiment = BVDExperiment(None, "squared", "param_name", [1, 0])
        self.bvd_experiment.n_trials = 5

    def test_generate_bootstrap(self):
        train_data = np.arange(100)
        train_data = train_data.reshape((-1, 2))
        bootstrap_indices = self.bvd_experiment._generate_bootstraps(train_data, replace=False, max_samples=1.0)
        self.assertEqual(train_data.shape,
                         train_data[bootstrap_indices[0], ...].shape)
        assert_allclose(train_data.var(),
                        train_data[bootstrap_indices[0], ...].var())


class FitModelTest(unittest.TestCase):

    def setUp(self):
        self.bvd_experiment = BVDExperiment(None, "squared", "param_name", [1, 0])

    def test_fit_model_without_progressive(self):
        ensemble = TestEnsemble()
        # Test with ensemble
        self.bvd_experiment.is_ensemble = True
        self.bvd_experiment.warm_start = False
        self.bvd_experiment._fit_model(ensemble, train_data=None,
                                       train_labels=None)

        self.assertEqual(ensemble.n_times_fit_called, 1)
        for model in ensemble.estimators_:
            self.assertEqual(model.n_times_fit_called, 0)

        self.bvd_experiment._fit_model(ensemble, train_data=None,
                                       train_labels=None)

        self.assertTrue(hasattr(ensemble, "estimators_"))
        self.assertEqual(ensemble.n_times_fit_called, 2)
        for model in ensemble.estimators_:
            self.assertEqual(model.n_times_fit_called, 0)

        # Test with individual model
        model = TestModel()
        self.bvd_experiment.is_ensemble = False
        self.bvd_experiment._fit_model(model, train_data=None,
                                       train_labels=None)
        self.assertEqual(model.n_times_fit_called, 1)

        self.bvd_experiment._fit_model(model, train_data=None,
                                       train_labels=None)
        self.assertEqual(model.n_times_fit_called, 2)

    def test_fit_with_progressive(self):
        ensemble = TestEnsemble()
        # Test with ensemble
        self.bvd_experiment.is_ensemble = True
        self.bvd_experiment.warm_start = True
        # TODO: _fit_model now has sample_weights parameter
        self.bvd_experiment._fit_model(ensemble, train_data=None,
                                       train_labels=None)

        self.assertEqual(ensemble.n_times_fit_called, 1)
        for model in ensemble.estimators_:
            self.assertEqual(model.n_times_fit_called, 0)

        self.assertTrue(hasattr(ensemble, "estimators_"))
        self.bvd_experiment._fit_model(ensemble, train_data=None,
                                       train_labels=None)

        self.assertEqual(ensemble.n_times_fit_called, 1)
        for model in ensemble.estimators_:
            self.assertEqual(model.n_times_fit_called, 1)

        # Test with individual model
        model = TestModel()
        self.bvd_experiment.is_ensemble = False
        self.bvd_experiment.warm_start = True

        self.bvd_experiment._fit_model(model,  train_data=None,
                                       train_labels=None)
        self.assertEqual(model.n_times_fit_called, 1)

        self.bvd_experiment._fit_model(model, train_data=None,
                                       train_labels=None)
        self.assertEqual(model.n_times_fit_called, 2)


class ParameterUpdateTest(unittest.TestCase):

    def setUp(self):
        self.bvd_experiment = BVDExperiment(None, "squared", "param_name", [1, 0])

    def test_param_update_ens_without_progressive(self):
        # Check ensemble parameters update correctly
        ensemble = TestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "ensemble_parameter"
        self.bvd_experiment.is_ensemble = True
        self.bvd_experiment.warm_start = False
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.ensemble_parameter, 2)
        self.assertFalse(hasattr(ensemble.base_estimator, "ensemble_parameter"))
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.ensemble_parameter, 7)

        # Check individual model parameter update correctly
        ensemble = TestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "model_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.model_parameter, 2)
        self.assertFalse(hasattr(ensemble, "model_parameter"))
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.base_estimator.model_parameter, 7)
        self.assertFalse(hasattr(ensemble, "model_parameter"))
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 7)

        # Check shared parameter update correctly
        ensemble = TestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "shared_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 2)
        self.assertEqual(ensemble.shared_parameter, 2)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 2)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.shared_parameter, 7)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 7)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 7)

    def test_param_update_ens_with_progressive(self):
        # Check ensemble parameters update correctly
        ensemble = TestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "ensemble_parameter"
        self.bvd_experiment.is_ensemble = True
        self.bvd_experiment.warm_start=True
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.ensemble_parameter, 2)
        self.assertFalse(hasattr(ensemble.base_estimator, "ensemble_parameter"))
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.ensemble_parameter, 5)

        # Check individual model parameter update correctly
        ensemble = TestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "model_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.model_parameter, 2)
        self.assertFalse(hasattr(ensemble, "model_parameter"))
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertFalse(hasattr(ensemble, "model_parameter"))
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 5)

        # Check shared parameter update correctly
        ensemble = TestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "shared_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 2)
        self.assertEqual(ensemble.shared_parameter, 2)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 2)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        # We don't call ensemble.fit() here because the estimators_ already exist, and calling fit will wipe them
        # (essentially, we are mimicking _fit_model())
        self.assertEqual(ensemble.shared_parameter, 5)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 5)

    def test_param_update_single_without_progressive(self):
        # Check ensemble parameters update correctly
        model = TestModel()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "model_parameter"
        self.bvd_experiment.is_ensemble = False
        self.bvd_experiment.warm_start = False
        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=0)
        self.assertEqual(model.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=1)
        self.assertEqual(model.model_parameter, 7)


    def test_param_update_single_with_progressive(self):
        # Check ensemble parameters update correctly
        model = TestModel()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "model_parameter"
        self.bvd_experiment.is_ensemble = False
        self.bvd_experiment.warm_start = True
        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=0)
        self.assertEqual(model.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=1)
        self.assertEqual(model.model_parameter, 5)

class SKParameterUpdateTest(unittest.TestCase):

    def setUp(self):
        self.bvd_experiment = BVDExperiment(None, "squared", "param_name", [1, 0])

    def test_param_update_ens_without_progressive(self):
        # Tests that parameters update correctly in sklearn models
        # Check ensemble parameters update correctly

        # Create an ensemble and update the ensemble parameter
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "ensemble_parameter"
        self.bvd_experiment.is_ensemble = True
        self.bvd_experiment.warm_start = False
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.ensemble_parameter, 2)
        # update parameter again
        self.assertFalse(hasattr(ensemble.base_estimator, "ensemble_parameter"))
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.ensemble_parameter, 7)
        self.assertFalse(hasattr(ensemble.base_estimator, "ensemble_parameter"))

        # Check individual model parameter update correctly
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "base_estimator__model_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.model_parameter, 2)
        self.assertFalse(hasattr(ensemble, "model_parameter"))
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.base_estimator.model_parameter, 7)
        self.assertFalse(hasattr(ensemble, "model_parameter"))
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 7)

        # Check shared parameter update correctly
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "shared_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 0)
        self.assertEqual(ensemble.shared_parameter, 2)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 0)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.shared_parameter, 7)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 0)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 0)

        # Check that shared parameter updates correctly in the ensemble members
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "base_estimator__shared_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.shared_parameter, 0)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 2)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 2)
        # Move on to second parameter
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.shared_parameter, 0)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 7)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 7)

    def test_param_update_ens_with_progressive(self):
        # Check ensemble parameters update correctly
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "ensemble_parameter"
        self.bvd_experiment.is_ensemble = True
        self.bvd_experiment.warm_start=True
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.ensemble_parameter, 2)
        self.assertFalse(hasattr(ensemble.base_estimator, "ensemble_parameter"))
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.ensemble_parameter, 5)

        # Check individual model parameter update correctly
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "base_estimator__model_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.model_parameter, 2)

        self.assertFalse(hasattr(ensemble, "model_parameter"))
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertFalse(hasattr(ensemble, "model_parameter"))
        # New parameter should be difference between the 1st and 2nd
        # values of parameter_values
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.model_parameter, 5)

        # Check shared parameter update correctly in ensemble
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "shared_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=0)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 0)
        self.assertEqual(ensemble.shared_parameter, 2)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 0)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        # We don't call ensemble.fit() here because the estimators_
        # already exist, and calling fit will wipe them
        # (essentially, we are mimicking _fit_model())
        self.assertEqual(ensemble.shared_parameter, 5)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 0)

        # Check shared parameter update correctly in individual members
        ensemble = SKLearnTestEnsemble()
        self.bvd_experiment.parameter_values = [2, 7]
        ensemble.fit(None, None)
        self.bvd_experiment.parameter_name = "base_estimator__shared_parameter"
        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        self.assertEqual(ensemble.base_estimator.shared_parameter, 0)
        self.assertEqual(ensemble.shared_parameter, 0)
        ensemble.fit(None, None)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 0)

        self.bvd_experiment._update_model_parameter(ensemble,
                                                    param_idx=1)
        # We don't call ensemble.fit() here because the estimators_
        # already exist, and calling fit will wipe them
        # (essentially, we are mimicking _fit_model())
        self.assertEqual(ensemble.shared_parameter, 0)
        for estimator in ensemble.estimators_:
            self.assertEqual(estimator.shared_parameter, 5)

    def test_param_update_single_without_progressive(self):
        # Check ensemble parameters update correctly
        model = SKLearnTestModel()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "model_parameter"
        self.bvd_experiment.is_ensemble = False
        self.bvd_experiment.warm_start = False
        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=0)
        self.assertEqual(model.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=1)
        self.assertEqual(model.model_parameter, 7)


    def test_param_update_single_with_progressive(self):
        # Check ensemble parameters update correctly
        model = SKLearnTestModel()
        self.bvd_experiment.parameter_values = [2, 7]
        self.bvd_experiment.parameter_name = "model_parameter"
        self.bvd_experiment.is_ensemble = False
        self.bvd_experiment.warm_start = True
        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=0)
        self.assertEqual(model.model_parameter, 2)

        self.bvd_experiment._update_model_parameter(model,
                                                    param_idx=1)
        self.assertEqual(model.model_parameter, 5)

class TripleNestedModelTest(unittest.TestCase):

    def setUp(self):
        base_estimator = SKLearnTestModel(nested_model=True)
        self.ensemble = SKLearnTestEnsemble(base_estimator=base_estimator)

    def test_warm_started_model(self):
        bvd_experiment = BVDExperiment(self.ensemble, "classify", "base_estimator__nested_model__model_parameter",
                                       parameter_values=[2, 7], warm_start=True)
        bvd_experiment._update_model_parameter(self.ensemble, param_idx=0)
        self.assertEqual(self.ensemble.base_estimator.nested_model.model_parameter, 2)
        self.ensemble.fit(None, None)
        for estimator in self.ensemble.estimators_:
            self.assertEqual(estimator.nested_model.model_parameter, 2)

        bvd_experiment._update_model_parameter(self.ensemble, param_idx=1)
        for estimator in self.ensemble.estimators_:
            estimator.fit(None, None)
        for estimator in self.ensemble.estimators_:
            self.assertEqual(estimator.nested_model.model_parameter, 5)

    def test_non_warm_started_model(self):
        bvd_experiment = BVDExperiment(self.ensemble, "classify","base_estimator__nested_model__model_parameter",
                                       parameter_values=[2, 7], warm_start=False)
        bvd_experiment._update_model_parameter(self.ensemble, param_idx=0)
        self.assertEqual(self.ensemble.base_estimator.nested_model.model_parameter, 2)
        self.ensemble.fit(None, None)
        for estimator in self.ensemble.estimators_:
            self.assertEqual(estimator.nested_model.model_parameter, 2)

        bvd_experiment._update_model_parameter(self.ensemble, param_idx=1)
        self.ensemble.fit(None, None)
        self.assertEqual(self.ensemble.base_estimator.nested_model.model_parameter, 7)
        for estimator in self.ensemble.estimators_:
            self.assertEqual(estimator.nested_model.model_parameter, 7)

class PredictionNoiseTest(unittest.TestCase):

    def setUp(self):
        self.bvd_experiment = BVDExperiment(None, "classify", "param_name", [1, 0])

    def test_pred_noise(self):
        pred = np.zeros([10, 3])
        pred[0, 0] = 1.
        noisy_pred = _add_model_smoothing(pred)
        first_entry = (1 - 1e-9) +  1e-9 * (1/3)
        other_entries = 1e-9 * 1/3
        entries = np.array([first_entry, other_entries, other_entries])
        assert_allclose(noisy_pred[0, :], entries)
        # The next test is no longer compatible with how noise is implemented
        # assert_allclose(noisy_pred[1, :], np.array([1/3., 1/3., 1/3.]))


class WarmStartDetectionTests(unittest.TestCase):

    def setUp(self):
        self.model = SKLearnTestModel()
        self.ensemble = SKLearnTestEnsemble()


    def test_warm_start(self):

        # If neither model nor base_estimator have warm_start, warm_start should agree with argument,
        # unless no argument given, then it should be false
        experiment_1 = BVDExperiment(self.model, "squared", "param_name", [])
        experiment_2 = BVDExperiment(self.ensemble, "squared", "param_name", [])
        self.assertEqual(experiment_1.warm_start, False)
        self.assertEqual(experiment_2.warm_start, False)

        experiment_3 = BVDExperiment(self.model, "squared", "param_name", [], warm_start=True)
        experiment_4 = BVDExperiment(self.ensemble, "squared", "param_name", [], warm_start= True)
        self.assertEqual(experiment_3.warm_start, True)
        self.assertEqual(experiment_4.warm_start, True)

        # If warm_start argument is not given, should be inferred from base_estimator first, then ensemble
        self.model.warm_start = True
        self.ensemble.warm_start = False
        self.ensemble.base_estimator.warm_start = True
        experiment_5 = BVDExperiment(self.model, "squared", "param_name", [])
        experiment_6 = BVDExperiment(self.ensemble, "squared", "param_name", [])
        self.assertEqual(experiment_5.warm_start, True)
        self.assertEqual(experiment_6.warm_start, True)

        # If warm_start argument is given, it should override the argument from base_estimator and from the model
        self.model.warm_start = True
        self.ensemble.warm_start = True
        self.ensemble.base_estimator.warm_start = True
        experiment_7 = BVDExperiment(self.model, "squared", "param_name", [], warm_start=False)
        experiment_8 = BVDExperiment(self.ensemble, "squared", "param_name", [], warm_start=False)
        self.assertEqual(experiment_7.warm_start, False)
        self.assertEqual(experiment_8.warm_start, False)

class LogisticLossTests(unittest.TestCase):

    def setUp(self):

        self.labels = np.ones(11)
        self.preds = 3 * np.ones([5, 2, 11])
        self.preds[:, 1, :] = self.preds[:, 1, :] * -1

    def test_decomposition(self):


        self.decomp = LogisticMarginLoss(self.preds, self.labels, is_additive=False)
        def logistic(x):
            return np.log(1. + np.exp(-x))

        assert_allclose(self.decomp.expected_ensemble_loss.mean(), np.log(2))
        assert_allclose(self.decomp.average_bias.mean(), 0.5 * (logistic(3) + logistic(-3)))
        assert_allclose(self.decomp.diversity.mean(), 0.5 * (logistic(3) + logistic(-3)) - np.log(2))
        assert_allclose(self.decomp.average_variance.mean(), 0.)


    def test_boosting_decomposition(self):

        def logistic(x):
            return np.log(1. + np.exp(-x))

        self.boosting_decomp = LogisticMarginLoss(self.preds, self.labels, is_additive=True)
        assert_allclose(self.boosting_decomp.expected_ensemble_loss.mean(), np.log(2))
        assert_allclose(self.boosting_decomp.average_bias.mean(), 0.5 * (logistic(2 * 3) + logistic(2 * -3)))
        assert_allclose(self.boosting_decomp.diversity.mean(), 0.5 * (logistic(2 * 3) + logistic(2 * -3)) - np.log(2))
        assert_allclose(self.boosting_decomp.average_variance.mean(), 0.)

class WeightedModeTests(unittest.TestCase):

    def setUp(self):
        self.x = np.random.choice(10, (15, 21, 100))

    def test_unweighted_mode(self):
        weighted_result = mode_with_random_ties(self.x, axis=0, random_ties=False)
        unweighted_result = unweighted_mode(self.x, axis=0)
        assert_allclose(weighted_result, unweighted_result)

        weighted_result = mode_with_random_ties(self.x, axis=1, random_ties=False)
        unweighted_result = unweighted_mode(self.x, axis=1)
        assert_allclose(weighted_result, unweighted_result)

class ZeroOneLossTests(unittest.TestCase):

    def setUp(self):
        self.preds1 = np.zeros([3, 4, 1], dtype=int)

        self.preds1[0, :, 0] = np.array([2, 1, 2, 3])
        self.preds1[1, :, 0] = np.array([1, 2, 3, 1])
        self.preds1[2, :, 0] = np.array([1, 1, 2, 3])

        self.label = np.array([1], dtype=int)

        self.decomp = ZeroOneLoss(self.preds1, self.label)

    def test_decomposition_terms(self):
        calculated_diversity_effect = (self.preds1 != self.label).mean()
        calculated_diversity_effect = calculated_diversity_effect - self.decomp.expected_ensemble_loss.mean()
        assert_allclose(self.decomp.diversity_effect, calculated_diversity_effect)

        calculated_variance_effect = (self.preds1 != self.label).mean()
        calculated_variance_effect = calculated_variance_effect - self.decomp.average_bias
        assert_allclose(self.decomp.average_variance_effect, calculated_variance_effect)

        assert_allclose(self.decomp.expected_ensemble_loss.mean(), 1./3.)
        assert_allclose(self.decomp.average_bias.mean(), 0.5)

    def test_decomposition_terms_random(self):
        for idx in range(100):
            ensemble_size = np.random.choice(20) + 1
            n_trials = np.random.choice(20) + 1
            n_classes = np.random.choice([2, 3, 4, 9])
            n_points = np.random.choice(100) + 1

            preds = np.random.choice(n_classes, size=[n_trials, ensemble_size, n_points])
            labels = np.random.choice(n_classes, n_points)

            decomp = ZeroOneLoss(preds, labels)

            calculated_diversity_effect = (preds != labels).mean()
            calculated_diversity_effect = calculated_diversity_effect - decomp.expected_ensemble_loss.mean()
            assert_allclose(decomp.diversity_effect.mean(), calculated_diversity_effect, atol=1e-15)

            calculated_variance_effect = (preds != labels).mean()
            calculated_variance_effect = calculated_variance_effect - decomp.average_bias.mean()
            assert_allclose(decomp.average_variance_effect.mean(), calculated_variance_effect, atol=1e-15)


class WeightedZeroOneLossTests(unittest.TestCase):

    def setUp(self):
        self.weights = np.zeros((2, 3, 1))
        self.preds = np.zeros((2, 3, 1), dtype=int)
        self.class_one_label = np.array([1], dtype=int)
        self.class_two_label = np.array([2], dtype=int)

        self.weights[:, :, 0] = np.array([[1, 2, 3], [4, 5, 6]])
        self.preds[:, :, 0] = np.array([[2, 1, 2], [1, 2, 1]])

    def test_average_bias(self):
        for random_ties in [True, False]:
            class_one_decomp = ZeroOneLoss(self.preds, self.class_one_label, self.weights, random_ties=random_ties)
            hand_calculated_class_one = 0.5 * (2. / 6. + 5. / 15)
            assert_allclose(hand_calculated_class_one, class_one_decomp.average_bias.mean())

            class_two_decomp = ZeroOneLoss(self.preds, self.class_two_label, self.weights, random_ties=random_ties)
            hand_calculated_class_two = 0.5 * (4. / 6. + 10. / 15.)
            assert_allclose(hand_calculated_class_two, class_two_decomp.average_bias.mean())

    def test_average_variance_effect(self):
        for random_ties in [True, False]:
            class_one_decomp = ZeroOneLoss(self.preds, self.class_one_label, self.weights, random_ties=random_ties)
            hand_calculated_class_one_p1 = 0.5 * (4. / 6. + 5. / 15)
            hand_calculated_class_one_p2 = 0.5 * (2. / 6. + 5. / 15)
            hand_calculated_class_one = hand_calculated_class_one_p1 - hand_calculated_class_one_p2
            assert_allclose(hand_calculated_class_one, class_one_decomp.average_variance_effect.mean())

            class_two_decomp = ZeroOneLoss(self.preds, self.class_two_label, self.weights, random_ties=random_ties)
            hand_calculated_class_two_p1 = 0.5 * (2. / 6. + 10. / 15.)
            hand_calculated_class_two_p2 = 0.5 * (4. / 6. + 10. / 15.)
            hand_calculated_class_two = hand_calculated_class_two_p1 - hand_calculated_class_two_p2
            assert_allclose(hand_calculated_class_two, class_two_decomp.average_variance_effect.mean())

    def test_diversity_effect(self):
        for random_ties in [True, False]:
            class_one_decomp = ZeroOneLoss(self.preds, self.class_one_label, self.weights, random_ties=random_ties)
            hand_calculated_class_one_p1 = 0.5 * (4. / 6. + 5. / 15)
            hand_calculated_class_one_p2 = 0.5
            hand_calculated_class_one = hand_calculated_class_one_p1 - hand_calculated_class_one_p2
            assert_allclose(hand_calculated_class_one, class_one_decomp.diversity_effect.mean())

            class_two_decomp = ZeroOneLoss(self.preds, self.class_two_label, self.weights, random_ties=random_ties)
            hand_calculated_class_two_p1 = 0.5 * (2. / 6. + 10. / 15.)
            hand_calculated_class_two_p2 = 0.5
            hand_calculated_class_two = hand_calculated_class_two_p1 - hand_calculated_class_two_p2
            assert_allclose(hand_calculated_class_two, class_two_decomp.diversity_effect.mean())


if __name__ == '__main__':
    unittest.main()
