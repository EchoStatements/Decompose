"""
The decomposition class provides general functionality for
taking an array of the outputs of the ensemble members,
computing the expected error, decomposing that error into ensemble bias
and variance terms, and further decomposing those into ensemble member
bias, variance, and ambiguity terms.

See the :ref:`Decompositions module<decompose>`
for how to compute decomposition_class for specific conditional target distributions.

In the following, :math:`y` denotes the ground truth label,
:math:`q_i` an ensemble member model (itself a random variable
depending on training data), :math:`\\bar{q}` an ensemble model (
likewise a random variable). A star, e.g., :math:`\\bar{q}^*` denotes
a central model with respect to training data. :math:`B(\\cdot, \\cdot)`
denotes the Bregman divergence. This notation is taken from (https://arxiv.org/abs/2301.03962v1) .
"""

import numpy as np
from cached_property import cached_property

class BregmanDecomposition(object):
    """
    A base class for decomposition classes based on Bregman divergences.

    This class also provides the base class for the MarginDecomposition,

    See the subclasses in bregman_losses and margin_losses for a description of the inputs.

    Parameters
    ----------
    pred : ndarray of shape (n_trials, n_ensemble_members, n_test_examples) or (n_trials, n_ensemble_members, n_test_examples, n_classes)
        Predictions to be used in the decomposition.
    labels : ndarray of shape (n_test_examples)
        Ground truth labels.

    Attributes
    ----------
    etas : ndarray of same shape as preds
        Transformation of predictions into dual coordinates via ::math::`eta = [\grad \phi](q)`
    n_training_data_resamples : int
        Number of data resamples that are averaged over to approximate the expectation
    n_ensemble_members : int
        Number of ensemble members
    n_test_examples : int
        Number of examples

    See Also
    --------
    SquaredLoss, MultivariateSquaredLoss, BinaryCrossEntropy, CrossEntropy,
    PoissonLoss

    """
    def __init__(self, pred, labels):
        if pred.shape[2] != labels.shape[0]:
            raise ValueError('Third dimension of model predictions should match '
            'first dimension of labels (# test data points)')

        self.pred = pred
        self.labels = labels

        # Compute the dual coordinates
        self.etas = self._generator_gradient()

        self.n_training_data_resamples = pred.shape[0]
        self.n_ensemble_members = pred.shape[1]
        self.n_test_examples = pred.shape[2]

    def _bregman_divergence(self, p, q):
        if len(p.shape) != len(q.shape):
            raise ValueError(f"p and q should be the same shape, are {p.shape} and {q.shape}")
        if len(p.shape) == 3:
            grad_term = self._generator_gradient(q) * (p - q)
        else:
            grad_term = np.einsum("ijkl,ijkl->ijk", self._generator_gradient(q), p - q)
        return self.bregman_generator(p) - self.bregman_generator(q) - grad_term

    def _bregman_expectation(self, axes1=(), axes2=(), axes3=()):
        centroid = self._inverse_generator_gradient(
            self.etas.mean(axis=axes1, keepdims=True)
        )
        individuals = self._inverse_generator_gradient(
            self.etas.mean(axis=axes2, keepdims=True)
        )
        return self._bregman_divergence(centroid, individuals).mean(axis=axes3).squeeze()


    def _error_function(self, axes1=(), axes2=()):
        # For each of the expected error/bias terms in the double decomposition_class we:
        # 1) Average the outputs over one or more axes;
        # 2) Apply the gradient function for the generator ;
        # 3) Average over zero or more axes.
        #
        # This function generalizes this, allowing it to be used to compute each
        # of the expected error and bias terms.

        x = self._inverse_generator_gradient(
            self.etas.mean(axis=axes1, keepdims=True))

        error = self._compute_error(x, self.labels)
        return np.mean(error, axis=axes2).squeeze()

    @cached_property
    def expected_ensemble_loss(self):
        """
        Compute the expected ensemble loss (with respect to training datasets)
        for each test data point, :math:`\\mathbb{E}_D[ B (y, \\bar{q})]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point

        """
        return self._error_function(1, 0)

    @cached_property
    def expected_member_loss(self):
        """
        Compute the expected average model loss for each test data point,
        :math:`\\mathbb{E}_D[ \\frac{1}{M} \\sum_{i=1}^M B(y, q_i)]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point

        """
        return self._error_function(axes2=(0, 1))


    @cached_property
    def ensemble_bias(self):
        """
        Compute the loss of the central ensemble for each test data point
        This is equal to the bias of the central ensemble plus the noise
        :math:`B(y, \\bar{q}^*)`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point

        """
        return self._error_function((0, 1))

    @cached_property
    def ensemble_variance(self):
        """
        Compute the ensemble variance for each test data point,
        :math:`\\mathbb{E}_D[B(\\bar{q}^*, \\bar{q})]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point

        """
        return self._bregman_expectation((0, 1), (1), 0)

    @cached_property
    def average_bias(self):
        """
        Compute the average loss of the central models for each test data point,
        This is equal to the average bias of the central models plus the conditional entropy
        :math:`\\frac{1}{M} \\sum_{i=1}^M B(y, q_i^*) + H(Y|X)`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point

        """
        return self._error_function(0, 1)

    @cached_property
    def average_variance(self):
        """
        Compute the average model variance for each test data point,
        :math:`\\frac{1}{M} \\sum_{i=1}^M \\mathbb{E}_D[B(q_i^*, q_i)]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point

        """
        return self._bregman_expectation(0, (), (0, 1))

    @cached_property
    def disparity(self):
        """
        Compute the disparity for each test data point,
        :math:`\\frac{1}{M} \\sum_{i=1}^M B(\\bar{q}^*, q_i^*)`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point

        """
        return self._bregman_expectation((0,1), 0, (1))

    @cached_property
    def diversity(self):
        """
        Compute the diversity for each test data point,
        :math:`\\mathbb{E}_D[\\frac{1}{M} \\sum_{i=1}^M B(\\bar{q}, q_i)]`.

        Returns
        -------
        array_like
            A (N,)-shape array with one value per test data point
        """
        return self._bregman_expectation(1, (), (0,1))

    def _non_centroid_error_function(self, combination_func, axes1=(), axes2=()):

        x = combination_func(self.pred, axis=axes1, keepdims=True)

        error = self._compute_error(x, self.labels)
        return np.mean(error, axis=axes2).squeeze()

    def _non_centroid_bregman_expectation(self, combination_func, axes1=(), axes2=(), axes3=()):
        q_dagger = combination_func(self.pred, axis=axes1, keepdims=True)
        # We probably never need axes2 != (), but doesn't hurt to have the option
        individuals = self._inverse_generator_gradient(
            self.etas.mean(axis=axes2, keepdims=True)
        )
        return self._bregman_divergence(q_dagger, individuals).mean(axis=axes3).squeeze()

    def general_target_dependent_term(self, combination_func, axes1=(), axes2=()):
        q_dagger = combination_func(self.pred, axis=axes1, keepdims=True)
        q_ast = self._inverse_generator_gradient(
            self.etas.mean(axis=axes1, keepdims=True))
        term_1 = self._generator_gradient(q_dagger) - self._generator_gradient(q_ast)
        if len(self.labels.shape) != len(self.pred.shape) - 2:
            labels = np.zeros((self.labels.size, np.max(self.labels) + 1))
            for idx in range(self.labels.shape[0]):
                labels[idx, self.labels[idx]] = 1
        else:
            labels = self.labels

        term_2 = labels - q_dagger
        if len(self.pred.shape) == 3:
            result = (term_1 * term_2).mean(axis=axes2).squeeze()
        else:
            result = np.einsum("ijkl,ijkl->ijk", term_1, term_2).mean(axis=axes2).squeeze()
        return result

    def target_dependent_term(self, combination_func):
        return self.general_target_dependent_term(combination_func, 1, 0)

    def diversity_like(self, combination_func):
        return self._non_centroid_bregman_expectation(combination_func, 1, (), (0,1))

    def non_centroid_expected_ensemble_risk(self, combination_func):
        return self._non_centroid_error_function(combination_func, 1, 0)


class MarginDecomposition(BregmanDecomposition):
    """
    Margin decomposition is to be considered a work-in-progress feature.
    """

    def _bregman_expectation(self, axes1=(), axes2=(), axes3=()):
        """
        The primary difference between the MarginDecomposition and the BregmanDecomposition
        is the form of the Bregman divergences and the centroids. Both of these can be handled by
        simply redefining the form of the Bregman Expectation

        Parameters
        ----------

        Returns
        -------

        """
        centroid = self.pred.mean(axis=axes1, keepdims=True) * self.labels
        individuals = self.pred.mean(axis=axes2, keepdims=True) * self.labels

        # For margin losses with generator $\ell$, arguments of the Bregman are swapped
        return self._bregman_divergence(individuals, centroid).mean(axis=axes3).squeeze()

    def _error_function(self, axes1=(), axes2=()):
        # For each of the expected error/bias terms in the double decomposition_class we:
        # 1) Average the outputs over one or more axes;
        # 2) Apply the inverse gradient function;
        # 3) Average over zero or more axes.
        #
        # This function generalizes this, allowing it to be used to compute each
        # of the expected error and bias terms.
        x = self.pred.mean(axis=axes1)
        error = self._compute_error(x, self.labels)
        return np.mean(error, axis=axes2).squeeze()

class EffectDecomposition(object):
    """
    Variation of the bias-variance-diversity decompose object using the effect decompose of James & Hastie.

    Each class inheriting from decomposition should implement a `_compute_error` function, which gives the error of
    predictions (first argument) given the true labels (second argument), and an aggregator function, which gives a
    method of combining a set of predictions into a single prediction (See ZeroOneLoss for an example).

    Parameters
    ----------
    pred : ndarray of shape (n_trials, ensemble_size, n_examples)
    labels : ndarray of shape (n_examples)
    """

    def __init__(self, pred, labels):
        self.pred = pred
        self.labels = labels

    def error_function(self, axes1=(), axes2=()):
        if axes1 != ():
            x = self.aggregator(self.pred, axis=axes1, weights=self.weights)
        else:
            x = self.pred
        error = self._compute_error(x, self.labels)
        if self.weights is not None:
            weights = np.mean(self.weights / self.weights.mean(axis=1, keepdims=True), axis=axes1, keepdims=True)
            assert weights.shape == error.shape
            error = weights * error
        return error.mean(axis=axes2).squeeze()

    def _central_model_difference(self, axes1=(), axes2=(), axes3=()):
        """
        Function to compute effect decomposition terms. The three axes parameters dictate which dimensions different
        aggregation operations occur.

        As an example, consider the weighted average variance-effect (Theorem 22 in
        the first ArXiV version of the paper). This quantity is made up of two terms $\frac{1}{M}\sum_i \mathbb{E}_D[a_i L(y, q_i)]$ and
        $\frac{1}{M} \sum_i L(Y, q_i^*)$. axes1 tells us the aggregation operation used in the central predictions
        (in this case the $q_i^*$). Since we are aggregating across trials (to get from individual $q_i$ to their centroid
        $q_i^*$) we set axes1=(0), since the 0th axes of self.pred is the one indexing trials. axes2 tells us what aggregation
        operation, if any needs to be performed to get the individual models in the first term. For the average variance
        effect, none is needed since individual models $q_i$ appear in the first term, so axes2=().
        Finally, we note that both terms are averaged over trials and ensemble members, this is dealt with by axes3. Setting
        axes3=(0, 1) averages over the dimensions corresponding to trials (giving $\mathbb{E}_D$) and ensemble members (giving $\frac{1}{M} \sum_i$)

        Parameters
        ----------
        axes1 : int or tuple of ints
            The dimension(s) along which the aggregation rule is applied to get the central model in the second term of the expression
        axes2 : int or tuple of ints
            The dimension(s) along which the aggregation rule is applied to get the individual models in the first term of the expression
        axes3 : int or tuple of ints
            The dimension(s) along which the final result is to be averaged

        Returns
        -------
        different : ndarray
            The difference in loss when going from the individual models to the central model

        """
        central_prediction = self.aggregator(self.pred, axis=axes1, weights=self.weights)
        individuals = self.aggregator(self.pred, axis=axes2)

        first_term =  self._compute_error(self.labels, individuals)
        second_term = self._compute_error(self.labels, central_prediction)
        if self.weights is not None:
            weights_for_first_term = np.mean(self.weights / self.weights.mean(axis=1, keepdims=True), axis=axes2, keepdims=True)
            weights_for_second_term = np.mean(self.weights / self.weights.mean(axis=1, keepdims=True), axis=axes1, keepdims=True)
            assert first_term.shape == weights_for_first_term.shape
            assert second_term.shape == weights_for_second_term.shape
            first_term = weights_for_first_term * first_term
            second_term = weights_for_second_term * second_term
        difference = first_term - second_term
        return difference.mean(axis=axes3)

    @cached_property
    def expected_ensemble_loss(self):
        """
        The expected loss of the ensemble on the test points, :math:`\\mathbb{E}[L(y, \\overline{f})]`

        Returns
        -------
        loss : ndarray
            Array of size (n_test_points, )

        """
        return self.error_function(1, 0)

    @cached_property
    def expected_member_loss(self):
        """
        The average expected loss of the ensemble members, :math:`\\frac{1}{M} \\sum_{i=1}^M \\mathbb{E} [L(y, f_i)]`

        Returns
        -------
        loss : ndarray
            Array of size (n_test_points, )
        """
        return self.error_function((), (0, 1))

    @cached_property
    def ensemble_bias(self):
        """
        The bias of the ensemble (i.e., the loss of the ensembles central prediction) :math:`L(y, \\overline{f}^\ast)`.

        Returns
        -------
        ensemble_bias : ndarray
            Array of size (n_test_points, )

        """
        return self.error_function((1, 0), ())

    @cached_property
    def average_bias(self):
        """
        The average bias of the ensemble members (i.e., the average loss of the member's central predictions),
        :math:`\\frac{1}{M} \\sum_{i=1}^M L(y, f_i^\\ast)`


        Returns
        -------
        average_bias : ndarray
            Array of size (n_test_points, )

        """
        return self.error_function(0, 1)

    @cached_property
    def ensemble_variance_effect(self):
        """
        The ensemble variance effect (i.e., the difference between the expected loss of the ensemble and the
        loss of the ensemble central prediction),
        :math:`\\mathbb{E}[L(y, \overline{f})] - L(y, \overline{f}^\\ast)`.

        Returns
        -------
        average_variance_effect : ndarray
            Array of size (n_test_points)

        """
        return self._central_model_difference((1, 0), 1, 0)

    @cached_property
    def average_variance_effect(self):
        """
        The average variance effect of the ensemble members (i.e., the difference between the average of the expected
        member losses and the average of the losses of the central models),
        :math:`\\frac{1}{M} \\sum_{i=1}^M \\mathbb{E}[L(y, f_i)] - \\frac{1}{M} \\sum_{i=1}^M L(y, f_i^\\ast)`.

        Returns
        -------
        average_variance_effect : ndarray
            Array of size (n_test_points)

        """
        return self._central_model_difference(0, (), (0, 1))

    @cached_property
    def diversity_effect(self):
        """
        The diversity-effect  for the ensemble (i.e., the expected difference between the average loss and the
        loss of the ensemble,
        :math:`\\frac{1}{M} \\sum_{i=1}^M \\mathbb{E}[L(y, f_i)] - \\mathbb{E}[ L(y, \\overline{f})]`.

        Returns
        -------
        diversity_effect : ndarray
            Array of size (n_test_points)

        """
        return self._central_model_difference(1, (), (0, 1))
