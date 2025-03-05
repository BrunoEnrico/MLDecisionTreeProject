import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


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
    def get_decision_tree_classifier(**kwargs) -> DecisionTreeClassifier:
        """
        Obtains the tree object for Machine Learning training.

        :return: Object ready for training
        """
        return DecisionTreeClassifier(**kwargs)

    @staticmethod
    def fit_decision_tree(decision_tree, feature_train: np.ndarray,
                          target_train: np.ndarray):
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
    def get_tree_score(target_test: np.ndarray, target_predict: DecisionTreeClassifier) -> float | int:
        """
        Returns the score of a tree prediction vs. real data

        :param target_test: Array containing the real data for comparison
        :param target_predict: Tree object with the predicted data
        :return: Float or Int of the accuracy of the comparison
        """
        return accuracy_score(target_test, target_predict)

    @staticmethod
    def get_confusion_matrix(target_test: np.ndarray, target_predict: DecisionTreeClassifier):
        """
        Returns the matrix between the comparison of a trained model prediction and the real data

        :param target_test: Array containing the real data for comparison
        :param target_predict: Tree object with the predicted data
        :return:
        """
        return confusion_matrix(y_true=target_test, y_pred=target_predict)

    @staticmethod
    def get_precision_score(target_test: np.ndarray, target_predict: DecisionTreeClassifier):
        """
        Calculates the precision score of a prediction over a test data

        :param target_test: Test data for comparison
        :param target_predict: Trained Decision Tree
        :return: Calculated precision score
        """
        return precision_score(y_true=target_test, y_pred=target_predict)

    @staticmethod
    def get_recall_score(target_test: np.ndarray, target_predict: DecisionTreeClassifier):
        """
        Calculates the recall score of a prediction over a test data

        :param target_test: Test data for comparison
        :param target_predict: Trained Decision Tree
        :return: Calculated recall score
        """
        return recall_score(y_true=target_test, y_pred=target_predict)

    @staticmethod
    def get_random_forest_classifier(**kwargs) -> RandomForestClassifier:
        """
        Initiates an instance of a Random Forest Classifier

        :param kwargs:
        :return: Instance of Random Forest Classifier
        """
        return RandomForestClassifier(**kwargs)

    @staticmethod
    def get_ada_boost_classifier(**kwargs):
        """
        Initiates an instance of Ada Boost Classifier

        :param kwargs:
        :return: Instance of Ada Boost Classifier
        """
        return AdaBoostClassifier(**kwargs)

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