from sklearn.datasets import *
import csv
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


def mease_dataset(n, d, J, q):
    """
    Function to generate examples from the synthetic dataset used in Mease (2008): Evidence Contrary to the Statistical View of Boosting

    Parameters
    ----------
    n : int
        number of examples to generate
    d : int
        number of features
    J : int
        number of "active" features
    q : float
        Bayes error rate

    Returns
    -------
    data: ndarray
    labels: ndarray

    """
    data = np.random.uniform(0, 1, size=[n, d])
    label_probs = q + (1. - (2. * q)) * (np.sum(data[:, :J], axis=1) > (J / 2.) )
    labels = np.random.uniform(0, 1, size=n)
    labels = (labels <= label_probs) * 1.
    return data, labels

def load_standard_dataset(dataset_name, frac_training, label_noise=0.,
                          normalize_data=False, random_state=None):
    """
    Function which returns standard datasets used in experiments. Some datasets have to be downloaded from the UCI database
    and put into the correct folders in order for the code to function. The location of the required directory is a `datasets`
    direct whose parent should be the directory that this file is contained in. The exact names of sub-directories for individual
    datasets can be inferred by looking at the `load_data` helper function (immediately below this function)

    Available datasets:
     - breast*
     - wine*
     - digits*
     - boston*
     - california*
     - friedman*
     - phoneme
     - gisette
     - HAR
     - isolet
     - mease*
     - cover
     - credit
     - ionosphere
     - spambase
     - semeion
     - mnist **
     - german-credit/south-german-credit
     - landsat

     * Datasets with an asterix work without requiring the user to first put the correct dataset in the
     datasets directory

     ** A script is available in the datasets directory to download and save mnist in the required form

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to be loaded
    frac_training: float, int
        The fraction of examples from the dataset to be used in the training set. If the value is a float, then it is taken
        to be the fraction of examples to be used (using `int(frac_training * num_examples)`. If the value is an int,
        then it is taken to be the number of examples to be included in the training set. In both cases, the remaining
        examples are all included in the test set
    label_noise: float
        For classification datasets, this is the proportion of examples which are randomly re-assigned to a different class.
        This is done using a uniform distribution over all classes, including the original class of the label.
    normalize_data: boolean (defaul=False)
        If true, the training and test data are normalised so that the training data has zero-mean and unit variance.
    random_state: int
        If given, this is the seed used for the random state when constructing the train/test split.

    Returns
    -------
        train_data : ndarray
            A numpy array of training data of size n_train_examples x n_features
        train_labels: ndarray
            A numpy array of training labels of size n_train_examples
        test_data : ndarray
            A numpy array of test data of size n_train_examples x n_features
        test_labels : ndarray
            A numpy array of test labels of size n_test_examples x n_features
    """
    data_and_labels = load_data(dataset_name)
    if len(data_and_labels) == 4:
        if frac_training is None:
            return data_and_labels
        else:
            data = np.concatenate((data_and_labels[0], data_and_labels[2]), axis=0)
            labels = np.concatenate((data_and_labels[1], data_and_labels[3]), axis=0)
    else:
        assert len(data_and_labels) == 2, "Length of list returned by load data " \
                                          f"should be 2 or 4, is {len(data_and_labels)}"
        data = data_and_labels[0]
        labels = data_and_labels[1]
    num_classes = np.unique(labels).shape[0]
    if isinstance(frac_training, int):
        num_training = frac_training
    else:
        num_training = int(frac_training * len(labels))

    rng = np.random.default_rng(seed=random_state)
    indices = rng.permutation(len(labels))
    train_indices = indices[:num_training]
    test_indices = indices[num_training:]
    train_data = data[train_indices, :]
    train_labels = labels[train_indices]
    test_data = data[test_indices, :]
    test_labels = labels[test_indices]

    if normalize_data:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

    if label_noise > 0.:
        train_labels = apply_classification_label_noise(train_labels, num_classes, label_noise)

    return train_data, train_labels, test_data, test_labels

def load_data(dataset_name, **params):
    """
    Load dataset using the dataset's name. This loads the raw datasets in for use in `load_standard_dataset`.
    and friedman.

    Parameters
    ----------
    dataset_name : string
        The name of the dataset to be loaded.

    Returns
    -------
    data : an (N, L) numpy array, where N is the number of examples in the dataset (including both train and test),
        and L is the number of features per example.
    labels : an (N,) numpy array, where N is the number of examples in the dataset (including both train and test).

    """

    data_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(data_dir, os.pardir)) + "/datasets/"
    # BREAST (2-class)
    if dataset_name == "breast":
        data, labels = load_breast_cancer(return_X_y=True)

    # WINE (3-class)
    elif dataset_name == "wine":
        data, labels = sklearn.datasets.load_wine(return_X_y=True)

    # DIGITS
    elif dataset_name == "digits":
        data, labels = sklearn.datasets.load_digits(return_X_y=True)

    # BOSTON
    elif dataset_name == "boston":
        data, labels = load_boston(return_X_y=True)

    # CALIFORNIA
    elif dataset_name == "california":
        cal_housing = fetch_california_housing()
        data, labels = cal_housing.data, cal_housing.target

    # FRIEDMAN
    elif dataset_name == "friedman":
        data, labels = make_friedman1(n_samples=1000, n_features=10,
                                      noise=0.3, random_state=None)

    elif dataset_name == "phoneme":
        all_data = np.loadtxt(data_dir + "phoneme.csv", delimiter=",")
        data = all_data[:, :-1]
        labels = all_data[:, -1].astype("int")

    elif dataset_name == "gisette":
        data = np.loadtxt(data_dir +  "gisette/gisette_train.data")
        labels = np.loadtxt(data_dir + "gisette/gisette_train.labels")
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(labels)
        labels = label_encoder.transform(labels)

    elif dataset_name.upper() == "HAR":
        this_dir = os.path.dirname(os.path.realpath(__file__))
        data1 = np.loadtxt(data_dir + "HAR/train/X_train.txt")
        data2 = np.loadtxt(data_dir + "HAR/test/X_test.txt")
        data = np.concatenate((data1, data2), axis=0)
        labels1 = np.loadtxt(data_dir + "HAR/train/y_train.txt")
        labels2 = np.loadtxt(this_dir + "HAR/test/y_test.txt")
        labels = np.concatenate((labels1, labels2), axis=0)
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(labels)
        labels = label_encoder.transform(labels)

    elif dataset_name == "isolet":
        data_dir = os.path.dirname(os.path.realpath(__file__))
        data1 = np.loadtxt(data_dir + "isolet/isolet1+2+3+4.data", delimiter=",")
        data2 =  np.loadtxt(data_dir + "isolet/isolet5.data", delimiter=",")
        data = np.concatenate((data1, data2), axis=0)
        labels = data[:, -1]
        data = data[:, :-1]
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(labels)
        labels = label_encoder.transform(labels)

    elif dataset_name == "mease":
        data, labels =  mease_dataset(1200, 20, 5, .1)

    elif dataset_name == "mease2":
        data, labels =  mease_dataset(1200, 20, 5, .2)

    elif dataset_name == "cover":
        dataset = sklearn.datasets.fetch_covtype()
        data = dataset["data"]
        labels = dataset["target"]

    elif dataset_name == "credit":
        data = np.loadtxt(data_dir + "/credit/default_credit.csv", skiprows=2, delimiter=",")
        labels = data[:, -1].astype(int)
        # remove target and consecutive ID numbers
        data = data[:, 1:-1]
        return data, labels

    elif dataset_name == "ionosphere":
        # from https://github.com/giswqs/Learning-Python/blob/master/Learning-Data-Mining-with-Python/Chapter%202/Ionosphere%20Nearest%20Neighbour.ipynb
        data_filename = data_dir + "ionosphere/ionosphere.data"
        data = np.zeros((351, 34), dtype='float')
        labels = np.zeros((351,), dtype='int')
        with open(data_filename, 'r') as input_file:
            reader = csv.reader(input_file)
            for i, row in enumerate(reader):
                # Get the data, converting each item to a float
                row_data = [float(datum) for datum in row[:-1]]
                # Set the appropriate row in our dataset
                data[i] = row_data
                # 1 if the class is 'g', 0 otherwise
                labels[i] = (row[-1] == 'g') * 1

    elif dataset_name == "spambase":
        data_file = data_dir + "spambase/spambase.data"
        data = np.loadtxt(data_file, delimiter=",")
        labels = data[:, -1].astype(int)
        # remove target and consecutive ID numbers
        data = data[:, :-1]
        return data, labels

    elif dataset_name == "semeion":
        dataset = sklearn.datasets.fetch_openml(data_id=1501)
        data = dataset["data"].to_numpy()
        labels = dataset["target"].to_numpy()

    elif dataset_name == "mnist":

        train_data = np.load(data_dir + "mnist_train_data.npy")
        test_data = np.load(data_dir + "mnist_test_data.npy")
        train_labels = np.load(data_dir + "mnist_train_labels.npy")
        test_labels = np.load(data_dir + "mnist_test_labels.npy")
        train_data = train_data.reshape((-1, 784))/255
        test_data = test_data.reshape((-1, 784))/255

        return train_data, train_labels, test_data, test_labels

    elif dataset_name in ["german_credit", "south_german_credit"]:
        data_file = data_dir + "credit/SouthGermanCredit.asc"
        data = np.loadtxt(data_file, delimiter=" ", skiprows=1)
        labels = data[:, -1].astype("int")
        data = data[:, :-1]
        return data, labels

    elif dataset_name == "landsat":
        data_file_1 = data_dir + "landsat/sat.trn"
        data_1 = np.loadtxt(data_file_1, delimiter=" ", skiprows=0)
        data_file_2 = data_dir + "landsat/sat.tst"
        data_2 =  np.loadtxt(data_file_2, delimiter=" ", skiprows=0)
        all_data = np.concatenate([data_1, data_2], axis=0).astype(float)
        labels = all_data[:, -1].astype(int)
        data = all_data[:, :-1]
        return data, labels


    else:
        raise KeyError("Data set name not known")

    return data, labels

def apply_classification_label_noise(labels, n_classes, change_prob):
    """
    Used to add label noise to datasets by randomly reassigning the labels of a randomly chosen subset of examples.
    Used by `load_standard_dataset` when the `label_noise` parameter is set to a non-zero value.

    Randomly reassigns class labels from some examples. Examples are chosen for reassignment with probability
    `change_prob`. Chosen examples are then reassigned according to a uniform distribution over [0,...,n_classes].
    This means that it is possible for an example to be reassigned to its original class.

    Parameters
    ----------
    labels : ndarray size (n_examples,)
        numpy array containing the labels on which classification noise is to be applied
    n_classes: int
        The number of classes in the data set. It is assumed that classes are labeled 0 to n-1
    change_prob : float
        Probability label is selected to be resampled

    Returns
    -------

    """
    if np.max(labels) > n_classes - 1 or np.min(labels) < 0:
        print("apply_classification_label_noise is intended to work with labels from 0 to n_classes-1.")
        print("Class label outside this range detected.")
        raise ValueError
    change_label = np.random.binomial(1, change_prob, labels.shape)
    random_labels = np.random.randint(low=0, high=n_classes, size=labels.shape)
    noisy_labels = labels * (1 - change_label) \
                   + random_labels * change_label
    return noisy_labels

if __name__ == "__main__":

    train_X, train_y, test_X, test_y =  load_standard_dataset("landsat", frac_training=0.75)
    print(f"spambase dataset loaded, train shape: {train_X.shape}")
    print(f"total samples: {train_X.shape[0] + test_X.shape[0]}")
    print(set(test_y))

    train_X, train_y, test_X, test_y =  load_standard_dataset("phoneme", frac_training=0.5)
    print(f"Phoneme dataset loaded, train shape: {train_X.shape}")
    print(f"total samples: {train_X.shape[0] + test_X.shape[0]}")
    print(set(test_y))
    train_X, train_y, test_X, test_y =  load_standard_dataset("digits", frac_training=0.5)
    print(f"Digits dataset loaded, train shape: {train_X.shape}")
    print(f"total samples: {train_X.shape[0] + test_X.shape[0]}")
    print(set(test_y))

    train_X, train_y, test_X, test_y =  load_standard_dataset("mnist", frac_training=None)
    print(f"total samples mnist: {train_X.shape[0] + test_X.shape[0]}")
    print(f"mnist train shape: {train_X.shape}")
    print(set(test_y))

    train_X, train_y, test_X, test_y =  load_data("mnist", frac_training=None)
    print(f"total samples mnist: {train_X.shape[0] + test_X.shape[0]}")
    print(f"mnist train shape: {train_X.shape}")
