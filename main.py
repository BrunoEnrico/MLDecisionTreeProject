import pandas as pd
from data_analysis import FraudAnalysis

class Main:
    def __init__(self):
        self.data = pd.read_csv("creditcard.csv")


    def process(self):
        fa = FraudAnalysis(data=self.data)
        print(f"Number of rows of the dataset: {fa.get_total_rows()}")
        print(f"Number of frauds present in the dataset: {fa.get_total_fraud('Class')}")
        print(f"Number of non-frauds present in the dataset: {fa.get_total_non_fraud('Class')}")
        print(f"Percentage of frauds present in the dataset: {fa.get_percentage_fraud('Class')}%")
        print(f"Percentage of non-frauds present in the dataset: {fa.get_percentage_non_fraud('Class')}%")


if __name__ == "__main__":
    main = Main()
    main.process()