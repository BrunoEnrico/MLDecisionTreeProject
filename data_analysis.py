import pandas as pd

class FraudAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_total_rows(self) -> int:
        """
        Eases the process of getting all the rows of a DataFrame

        :return: Number of rows of the given DataFrame
        """
        return len(self.data)

    def get_total_fraud(self, fraud_column: str) -> int:
        """
        Gets all frauds inside a DataFrane

        :param fraud_column: Target column of the model (binary columns only)
        :return: Number of frauds in the DataFrane
        """
        return self.data[fraud_column].sum()

    def get_total_non_fraud(self, fraud_column: str) -> int:
        """
        Gets the total non-fraud cases in a DataFrame

        :param fraud_column: Column containing the fraud cases (binary columns only)
        :return: Number of non-fraud cases
        """
        return self.get_total_rows() - self.get_total_fraud(fraud_column)

    def get_percentage_fraud(self, fraud_column: str, decimal: int = 2) -> float:
        """
        Gets the percentage of frauds in comparison to the total rows of the DataFrame

        :param fraud_column: Column containing the fraud cases (binary columns only)
        :param decimal: Number of decimal cases (default is 2)
        :return: Float percentage of the number of fraud cases
        """
        percentage = (self.get_total_fraud(fraud_column)/self.get_total_rows())*100
        return round(percentage, decimal)

    def get_percentage_non_fraud(self, fraud_column: str, decimal: int = 2) -> float:
        """
        Gets the percentage of non-fraud cases in a DataFrame

        :param fraud_column: Column containing the fraud cases (binary columns only)
        :param decimal: Number of decimal cases (default is 2)
        :return: Float percentage of the number of non-fraud cases
        """
        percentage = (self.get_total_non_fraud(fraud_column)/self.get_total_rows())*100
        return round(percentage, decimal)