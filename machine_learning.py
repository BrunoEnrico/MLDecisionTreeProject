import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


class MachineLearning:

    def __init__(self):
        pass

    @staticmethod
    def get_validator(**kwargs) -> StratifiedShuffleSplit:
        """
        Obtains the validator object to split datasets for Machine Learning training.

        :param kwargs: Parameters to initialize Stratified Shuffle Split class
        :return: Initialized Stratified Shuffle Split
        """
        return StratifiedShuffleSplit(**kwargs)

    @staticmethod
    def get_split(validator: StratifiedShuffleSplit, data: np.ndarray, target: np.ndarray) -> tuple:
        """
        Obtains the split of the dataset for Machine Learning training.

        :param validator: Stratified Shuffle Split class initialized
        :param data: Feature variables used to train the model (just the values)
        :param target: Dependent column
        :return: tuple containing the split
        """
        train_id, test_id = next(validator.split(X=data, y=target))
        return data[train_id], data[test_id], target[train_id], target[test_id]

    @staticmethod
    def get_decision_tree_classifier() -> DecisionTreeClassifier:
        """
        Obtains the tree object for Machine Learning training.

        :return: Object ready for training
        """
        return DecisionTreeClassifier()

    @staticmethod
    def fit_decision_tree(decision_tree: DecisionTreeClassifier, feature_train: np.ndarray,
                          target_train: np.ndarray) -> DecisionTreeClassifier:
        """
        Trains a decision tree object with a given feature and target data.

        :param decision_tree: Decision Tree object to train
        :param feature_train: Feature data to train the model
        :param target_train: Target data to train the model
        :return: Trained Decision Tree object
        """
        return decision_tree.fit(X=feature_train, y=target_train)

    @staticmethod
    def tree_predict(decision_tree: DecisionTreeClassifier, feature_test: np.ndarray) -> DecisionTreeClassifier:
        """

        :param decision_tree: Trained Decision Tree object
        :param feature_test: Feature to pass for the model to predict the target
        :return:
        """
        return decision_tree.predict(X=feature_test)

    @staticmethod
    def save_tree_plot(decision_tree: DecisionTreeClassifier, name: str, **kwargs) -> None:
        """

        :param decision_tree: Decision Tree object to plot
        :param name: Name to save the file
        :param kwargs: Arguments to pass for the 'plot_tree' function
        """
        plt.figure(figsize=(200, 100))
        tree.plot_tree(decision_tree=decision_tree, **kwargs)
        plt.savefig(name)
        plt.close()