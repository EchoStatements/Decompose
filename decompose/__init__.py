from decompose.decompositions import BregmanDecomposition, EffectDecomposition, MarginDecomposition

from decompose.bregman_losses import BinaryCrossEntropy, CrossEntropy, SquaredLoss, MultivariateSquaredLoss, PoissonLoss
from decompose.margin_losses import LogisticMarginLoss, ExponentialMarginLoss, BujaExponentialLoss, BujaLogisticMarginLoss
from decompose.zero_one_loss import ZeroOneLoss

from decompose.experiments import BVDExperiment
from decompose.plotting_utils import plot_bv, plot_bvd, plot_errors

__version__ = '0.1.0'
