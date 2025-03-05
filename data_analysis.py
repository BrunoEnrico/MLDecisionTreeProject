import pandas as pd

class FraudAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_total_rows(self) -> int:
        return len(self.data)

    def get_total_fraud(self, fraud_column: str) -> int:
        return self.data[fraud_column].sum()

    def get_total_non_fraud(self, fraud_column: str) -> int:
        return self.get_total_rows() - self.get_total_fraud(fraud_column)

    def get_percentage_fraud(self, fraud_column: str, decimal: int = 2) -> float:
        percentage = (self.get_total_fraud(fraud_column)/self.get_total_rows())*100
        return round(percentage, decimal)

    def get_percentage_non_fraud(self, fraud_column: str, decimal: int = 2) -> float:
        percentage = (self.get_total_non_fraud(fraud_column)/self.get_total_rows())*100
        return round(percentage, decimal)